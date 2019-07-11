# ==============================================================================
# Copyright (c) 2018 Rayhane Mama
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Tacotron2 modules.
The implementation of location sensitive attention is based on Rayhane-mamah's implementation.
Reference: https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/models/attention.py
"""

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.contrib.rnn import RNNCell, MultiRNNCell
from tensorflow.contrib.seq2seq import BahdanauAttention
from functools import reduce
from tacotron.modules import ZoneoutLSTMCell, Conv1d
from tacotron.rnn_wrappers import AttentionRNN
from tacotron.rnn_impl import LSTMImpl


class EncoderV2(tf.layers.Layer):

    def __init__(self, num_conv_layers, kernel_size, out_units, drop_rate,
                 zoneout_factor_cell, zoneout_factor_output, is_training,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(EncoderV2, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        assert out_units % 2 == 0
        self.out_units = out_units
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_output = zoneout_factor_output
        self.is_training = is_training
        self._lstm_impl = lstm_impl

        self.convolutions = [Conv1d(kernel_size, out_units, activation=tf.nn.relu, is_training=is_training,
                                    drop_rate=drop_rate,
                                    name=f"conv1d_{i}",
                                    dtype=dtype) for i in
                             range(0, num_conv_layers)]

    def call(self, inputs, input_lengths=None):
        conv_output = reduce(lambda acc, conv: conv(acc), self.convolutions, inputs)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            ZoneoutLSTMCell(self.out_units // 2, self.is_training, self.zoneout_factor_cell,
                            self.zoneout_factor_output, lstm_impl=self._lstm_impl, dtype=self.dtype),
            ZoneoutLSTMCell(self.out_units // 2, self.is_training, self.zoneout_factor_cell,
                            self.zoneout_factor_output, lstm_impl=self._lstm_impl, dtype=self.dtype),
            conv_output,
            sequence_length=input_lengths,
            dtype=inputs.dtype)
        return tf.concat(outputs, axis=-1)


def _location_sensitive_score(W_query, W_fill, W_keys, dtype=None):
    num_units = W_keys.shape[-1].value or tf.shape(W_keys)[-1]

    v_a = tf.get_variable("attention_variable",
                          shape=[num_units],
                          dtype=dtype,
                          initializer=tf.contrib.layers.xavier_initializer())
    b_a = tf.get_variable("attention_bias",
                          shape=[num_units],
                          dtype=dtype,
                          initializer=tf.zeros_initializer())

    return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fill + b_a), axis=[2])


class LocationSensitiveAttention(BahdanauAttention):

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length,
                 attention_kernel,
                 attention_filters,
                 smoothing=False,
                 cumulative_weights=True,
                 dtype=None,
                 name="LocationSensitiveAttention"):
        probability_fn = self._smoothing_normalization if smoothing else None

        super(LocationSensitiveAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            probability_fn=probability_fn,
            dtype=dtype,
            name=name)
        self._cumulative_weights = cumulative_weights
        self._dtype = dtype or backend.floatx()

        self.location_convolution = tf.layers.Conv1D(filters=attention_filters,
                                                     kernel_size=attention_kernel,
                                                     padding="SAME",
                                                     use_bias=True,
                                                     bias_initializer=tf.zeros_initializer(dtype=memory.dtype),
                                                     name="location_features_convolution",
                                                     dtype=dtype)

        self.location_layer = tf.layers.Dense(units=num_units,
                                              use_bias=False,
                                              name="location_features_layer",
                                              dtype=dtype)

    def __call__(self, query, state):
        previous_alignments = state
        with tf.variable_scope(None, "location_sensitive_attention", [query]):
            # processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
            processed_query = self.query_layer(query) if self.query_layer else query

            # -> [batch_size, 1, attention_dim]
            processed_query = tf.expand_dims(processed_query, 1)

            # [batch_size, max_time] -> [batch_size, max_time, 1]
            expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
            # location features [batch_size, max_time, filters]
            f = self.location_convolution(expanded_alignments)
            processed_location_features = self.location_layer(f)

            energy = _location_sensitive_score(processed_query, processed_location_features, self.keys,
                                               dtype=self._dtype)

        alignments = self._probability_fn(energy, state)
        if self._cumulative_weights:
            next_state = alignments + previous_alignments
        else:
            next_state = alignments
        return alignments, next_state

    def _smoothing_normalization(e):
        return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keep_dims=True)


class DecoderRNNV2(RNNCell):

    def __init__(self, out_units, attention_cell: AttentionRNN,
                 is_training, zoneout_factor_cell=0.0, zoneout_factor_output=0.0,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(DecoderRNNV2, self).__init__(name=name, trainable=trainable, **kwargs)

        self._cell = MultiRNNCell([
            attention_cell,
            ZoneoutLSTMCell(out_units, is_training, zoneout_factor_cell, zoneout_factor_output, lstm_impl=lstm_impl,
                            dtype=dtype),
            ZoneoutLSTMCell(out_units, is_training, zoneout_factor_cell, zoneout_factor_output, lstm_impl=lstm_impl,
                            dtype=dtype),
        ], state_is_tuple=True)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state):
        return self._cell(inputs, state)


class PostNetV2(tf.layers.Layer):

    def __init__(self, out_units, num_postnet_layers, kernel_size, out_channels, is_training, drop_rate=0.5,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(PostNetV2, self).__init__(name=name, trainable=trainable, **kwargs)

        final_conv_layer = Conv1d(kernel_size, out_channels, activation=None, is_training=is_training,
                                  drop_rate=drop_rate,
                                  name=f"conv1d_{num_postnet_layers}",
                                  dtype=dtype)

        self.convolutions = [Conv1d(kernel_size, out_channels, activation=tf.nn.tanh, is_training=is_training,
                                    drop_rate=drop_rate,
                                    name=f"conv1d_{i}",
                                    dtype=dtype) for i in
                             range(1, num_postnet_layers)] + [final_conv_layer]

        self.projection_layer = tf.layers.Dense(out_units, dtype=dtype)

    def call(self, inputs, **kwargs):
        output = reduce(lambda acc, conv: conv(acc), self.convolutions, inputs)
        projected = self.projection_layer(output)
        summed = inputs + projected
        return summed
