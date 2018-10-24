# BSD 3-Clause License
#
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
""" RNNWrappers. """

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, RNNCell
from tensorflow.contrib.seq2seq import AttentionWrapper
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
        self._out_units = out_units
        self._cell = cell
        self.out_projection = tf.layers.Dense(out_units)
        self.stop_token_projectioon = tf.layers.Dense(1)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return (self._out_units, 1)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        mel_output = self.out_projection(output)
        stop_token = self.stop_token_projectioon(output)
        return (mel_output, stop_token), res_state


class AttentionRNN(RNNCell):

    def __init__(self, cell, prenets: Tuple[PreNet],
                 attention_mechanism,
                 trainable=True, name=None, **kwargs):
        super(AttentionRNN, self).__init__(name=name, trainable=trainable, **kwargs)
        attention_cell = AttentionWrapper(
            DecoderPreNetWrapper(cell, prenets),
            attention_mechanism,
            alignment_history=True,
            output_attention=False)
        concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)
        self._cell = concat_cell

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
