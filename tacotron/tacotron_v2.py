import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention
from typing import Tuple
from tacotron.modules import PreNet
from tacotron.rnn_wrappers import AttentionRNN


# https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/models/attention.py
def _location_sensitive_score(W_query, W_fill, W_keys):
    dtype = W_query.dtype
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
                 cumulate_weights=True,
                 name="LocationSensitiveAttention"):
        probability_fn = self._smoothing_normalization if smoothing else None

        super(LocationSensitiveAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            probability_fn=probability_fn,
            name=name)
        self._cumulate_weights = cumulate_weights

        self.location_convolution = tf.layers.Conv1D(filters=attention_filters,
                                                     kernel_size=attention_kernel,
                                                     padding="SAME",
                                                     use_bias=True,
                                                     bias_initializer=tf.zeros_initializer(),
                                                     name="location_features_convolution")

        self.location_layer = tf.layers.Dense(units=num_units,
                                              use_bias=False,
                                              dtype=tf.float32,
                                              name="location_features_layer")

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

            energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)

        alignments = self._probability_fn(energy, state)
        if self._cumulate_weights:
            next_state = alignments + previous_alignments
        else:
            next_state = alignments
        return alignments, next_state

    def _smoothing_normalization(e):
        return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keep_dims=True)


def AttentionRNNV2(num_units,
                   prenets: Tuple[PreNet],
                   memory, memory_sequence_length,
                   attention_kernel,
                   attention_filters,
                   smoothing=False,
                   cumulate_weights=True):
    attention_mechanism = LocationSensitiveAttention(num_units, memory, memory_sequence_length,
                                                     attention_kernel, attention_filters, smoothing, cumulate_weights)
    return AttentionRNN(num_units, prenets, attention_mechanism)
