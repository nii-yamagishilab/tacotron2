# Copyright (c) 2017 Keith Ito
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
""" Basic building blocks.
PreNet, HighwayNet and Conv1d is modified from keithito's implementation.
Reference: https://github.com/keithito/tacotron/blob/master/models/modules.py

ZoneoutLSTMCell is adopted from Rayhane-mamah's implementation.
Reference: https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/models/modules.py

We are requesting to include an open source licence of Zoneout implementation to teganmaharaj
https://github.com/teganmaharaj/zoneout/issues/8
"""

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from functools import reduce


class Embedding(tf.layers.Layer):

    def __init__(self, num_symbols, embedding_dim, index_offset=0,
                 trainable=True, name=None, **kwargs):
        super(Embedding, self).__init__(name=name, trainable=trainable, **kwargs)
        self._num_symbols = num_symbols
        self._embedding_dim = embedding_dim
        self.index_offset = tf.convert_to_tensor(index_offset, dtype=tf.int64)

    def build(self, _):
        self._embedding = self.add_variable("embedding", shape=[self._num_symbols, self._embedding_dim],
                                            dtype=tf.float32)
        self.built = True

    def call(self, inputs, **kwargs):
        with tf.control_dependencies([tf.assert_greater_equal(inputs, self.index_offset),
                                      tf.assert_less(inputs, self.index_offset + self._num_symbols)]):
            return tf.nn.embedding_lookup(self._embedding, inputs - self.index_offset)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self._embedding_dim])

class ExternalEmbedding(tf.layers.Layer): ## TODO(ec) should this go here?
    def __init__(self, fname, num_symbols, embedding_dim, index_offset=0,
                 trainable=False, name=None, **kwargs):
        super(ExternalEmbedding, self).__init__(name=name, trainable=trainable, **kwargs)
        self._fname = fname
        self._num_symbols = num_symbols
        self._embedding_dim = embedding_dim
        self.index_offset = tf.convert_to_tensor(index_offset, dtype=tf.int64)

    def build(self, _):
        self._embedding = self.load_embedding_from_file(self._fname)

    def call(self, inputs, **kwargs):
        with tf.control_dependencies([tf.assert_greater_equal(inputs, self.index_offset),
                                      tf.assert_less(inputs, self.index_offset + self._num_symbols)]):
            return tf.nn.embedding_lookup(self._embedding, inputs - self.index_offset)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self._embedding_dim])
                              
    def load_embedding_from_file(self, fname):  ## TODO(ec) should this go here or elsewhere?
        xvectors = {}
        min = 9999999999  ## please do not have a speaker ID larger than this
        max = -9999999999
        vecsize = 0

        f = open(fname, 'r')
        for line in f:
            parts = line.strip().split('  ')
            spkr = int(parts[0][1:])  ## assuming vctk 0.91 format with a letter on front
            xvec = [float(x) for x in parts[1].strip('[]').strip().split(' ')]
            if vecsize == 0:
                vecsize = len(xvec)
            if spkr < min:
                min = spkr
            if spkr > max:
                max = spkr
            xvectors[spkr] = xvec
            
        xv_table = []
        for i in range(min, max+1):
            if i in xvectors.keys():
                xv_table.append(xvectors[i])
            else:
                empty = [0 for x in range(0, vecsize)]  ## placeholder for skipped speaker IDs
                xv_table.append(empty)

        return tf.constant(xv_table)
                                

class PreNet(tf.layers.Layer):

    def __init__(self, out_units, is_training, drop_rate=0.5,
                 apply_dropout_on_inference=False,
                 trainable=True, name=None, **kwargs):
        super(PreNet, self).__init__(name=name, trainable=trainable, **kwargs)
        self.out_units = out_units
        self.drop_rate = drop_rate
        self.is_training = is_training
        self.apply_dropout_on_inference = apply_dropout_on_inference
        self.dense = tf.layers.Dense(out_units, activation=tf.nn.relu)

    def build(self, _):
        self.built = True

    def call(self, inputs, **kwargs):
        dense = self.dense(inputs)
        dropout = tf.layers.dropout(dense, rate=self.drop_rate, training=self.dropout_enabled)
        return dropout

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)

    @property
    def dropout_enabled(self):
        return self.is_training or self.apply_dropout_on_inference


class HighwayNet(tf.layers.Layer):

    def __init__(self, out_units,
                 h_kernel_initializer=None,
                 h_bias_initializer=None,
                 t_kernel_initializer=None,
                 t_bias_initializer=tf.constant_initializer(-1.0),
                 trainable=True, name=None, **kwargs):
        super(HighwayNet, self).__init__(name=name, trainable=trainable, **kwargs)
        self.out_units = out_units
        self.H = tf.layers.Dense(out_units, activation=tf.nn.relu, name="H",
                                 kernel_initializer=h_kernel_initializer,
                                 bias_initializer=h_bias_initializer)
        self.T = tf.layers.Dense(out_units, activation=tf.nn.sigmoid, name="T",
                                 kernel_initializer=t_kernel_initializer,
                                 bias_initializer=t_bias_initializer)

    def build(self, input_shape):
        with tf.control_dependencies([tf.assert_equal(self.out_units, input_shape[-1])]):
            self.built = True

    def call(self, inputs, **kwargs):
        h = self.H(inputs)
        t = self.T(inputs)
        return h * t + inputs * (1.0 - t)

    def compute_output_shape(self, input_shape):
        return input_shape


class Conv1d(tf.layers.Layer):

    def __init__(self, kernel_size, out_channels, activation, is_training,
                 use_bias=False,
                 drop_rate=0.0,
                 trainable=True, name=None, **kwargs):
        super(Conv1d, self).__init__(name=name, trainable=trainable, **kwargs)
        self.is_training = is_training
        self.activation = activation
        self.drop_rate = drop_rate
        self.conv1d = tf.layers.Conv1D(out_channels, kernel_size, use_bias=use_bias, activation=None, padding="SAME")

    def build(self, _):
        self.built = True

    def call(self, inputs, **kwargs):
        conv1d = self.conv1d(inputs)
        batch_normalization = tf.layers.batch_normalization(conv1d, training=self.is_training)
        output = self.activation(batch_normalization) if self.activation is not None else batch_normalization
        output = tf.layers.dropout(output, self.drop_rate, training=self.is_training)
        return output

    def compute_output_shape(self, input_shape):
        return self.conv1d.compute_output_shape(input_shape)


class CBHG(tf.layers.Layer):

    def __init__(self, out_units, conv_channels, max_filter_width, projection1_out_channels, projection2_out_channels,
                 num_highway, is_training,
                 trainable=True, name=None, **kwargs):
        half_out_units = out_units // 2
        assert out_units % 2 == 0
        super(CBHG, self).__init__(name=name, trainable=trainable, **kwargs)

        self.out_units = out_units

        self.convolution_banks = [
            Conv1d(kernel_size,
                   conv_channels,
                   activation=tf.nn.relu,
                   is_training=is_training,
                   name=f"conv1d_K{kernel_size}")
            for kernel_size in range(1, max_filter_width + 1)]
        self.maxpool = tf.layers.MaxPooling1D(pool_size=2, strides=1, padding="SAME")

        self.projection1 = Conv1d(kernel_size=3,
                                  out_channels=projection1_out_channels,
                                  activation=tf.nn.relu,
                                  is_training=is_training,
                                  name="proj1")

        self.projection2 = Conv1d(kernel_size=3,
                                  out_channels=projection2_out_channels,
                                  activation=tf.identity,
                                  is_training=is_training,
                                  name="proj2")

        self.adjustment_layer = tf.layers.Dense(half_out_units)

        self.highway_nets = [HighwayNet(half_out_units) for i in range(1, num_highway + 1)]

    def build(self, _):
        self.built = True

    def call(self, inputs, input_lengths=None, **kwargs):
        conv_outputs = tf.concat([conv1d(inputs) for conv1d in self.convolution_banks], axis=-1)

        maxpool_output = self.maxpool(conv_outputs)

        proj1_output = self.projection1(maxpool_output)
        proj2_output = self.projection2(proj1_output)

        # residual connection
        highway_input = proj2_output + inputs

        if highway_input.shape[2] != self.out_units // 2:
            highway_input = self.adjustment_layer(highway_input)

        highway_output = reduce(lambda acc, hw: hw(acc), self.highway_nets, highway_input)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            GRUCell(self.out_units // 2),
            GRUCell(self.out_units // 2),
            highway_output,
            sequence_length=input_lengths,
            dtype=tf.float32)

        return tf.concat(outputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.out_units])


class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, is_training, zoneout_factor_cell=0.0, zoneout_factor_output=0.0, state_is_tuple=True,
                 name=None):
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)

        if zm < 0. or zs > 1.:
            raise ValueError('One/both provided Zoneout factors are not in [0, 1]')

        self._cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=state_is_tuple, name=name)
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
        self.is_training = is_training
        self.state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        # Apply vanilla LSTM
        output, new_state = self._cell(inputs, state, scope)

        if self.state_is_tuple:
            (prev_c, prev_h) = state
            (new_c, new_h) = new_state
        else:
            raise NotImplementedError("non-tuple state is not implemented")

        # Apply zoneout
        keep_rate_cell = 1.0 - self._zoneout_cell
        keep_rate_output = 1.0 - self._zoneout_outputs
        if self.is_training:
            c = keep_rate_cell * tf.nn.dropout(new_c - prev_c, keep_prob=keep_rate_cell) + prev_c
            h = keep_rate_output * tf.nn.dropout(new_h - prev_h, keep_prob=keep_rate_output) + prev_h
        else:
            c = (1.0 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
            h = (1.0 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h

        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat([c, h], axis=1)

        return output, new_state
