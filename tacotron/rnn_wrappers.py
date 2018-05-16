import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from typing import Tuple
from functools import reduce
from tacotron.modules import PreNet


class DecoderPreNetWrapper(RNNCell):

    def __init__(self, cell: RNNCell, prenets: Tuple[PreNet]):
        super(DecoderPreNetWrapper, self).__init__()
        self._cell = cell
        self.prenets = prenets

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
        prenet_output = reduce(lambda acc, pn: pn(acc), self.prenets, inputs)
        return self._cell(prenet_output, state)


class ConcatOutputAndAttentionWrapper(RNNCell):

    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state


class OutputAndStopTokenWrapper(RNNCell):

    def __init__(self, cell, out_units):
        super(OutputAndStopTokenWrapper, self).__init__()
        self._cell = cell
        self.out_projection = tf.layers.Dense(out_units)
        self.stop_token_projectioon = tf.layers.Dense(1)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return (self._cell.output_size, 1)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        mel_output = self.out_projection(output)
        stop_token = self.stop_token_projectioon(output)
        return (mel_output, stop_token), res_state
