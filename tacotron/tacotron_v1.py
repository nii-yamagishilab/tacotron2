import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, RNNCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from tacotron.modules import PreNet, CBHG
from tacotron.rnn_wrappers import DecoderPreNetWrapper, ConcatOutputAndAttentionWrapper
from tacotron.helpers import InferenceHelper, TrainingHelper
from functools import reduce
from typing import Tuple


class EncoderV1(tf.layers.Layer):

    def __init__(self, is_training,
                 cbhg_out_units=256, conv_channels=128, max_filter_width=16,
                 projection1_out_channels=128,
                 projection2_out_channels=128,
                 num_highway=4,
                 prenet_out_units=(256, 128), drop_rate=0.5,
                 trainable=True, name=None, **kwargs):
        super(EncoderV1, self).__init__(name=name, trainable=trainable, **kwargs)
        self.prenet_out_units = prenet_out_units

        self.prenets = [PreNet(out_unit, is_training, drop_rate) for out_unit in prenet_out_units]

        self.cbhg = CBHG(cbhg_out_units,
                         conv_channels,
                         max_filter_width,
                         projection1_out_channels,
                         projection2_out_channels,
                         num_highway,
                         is_training)

    def build(self, input_shape):
        embed_dim = input_shape[2].value
        with tf.control_dependencies([tf.assert_equal(self.prenet_out_units[0], embed_dim)]):
            self.built = True

    def call(self, inputs, input_lengths=None, **kwargs):
        prenet_output = reduce(lambda acc, pn: pn(acc), self.prenets, inputs)
        cbhg_output = self.cbhg(prenet_output, input_lengths=input_lengths)
        return cbhg_output

    def compute_output_shape(self, input_shape):
        return self.cbhg.compute_output_shape(input_shape)


class AttentionRNNV1(RNNCell):

    def __init__(self, num_units, prenets: Tuple[PreNet],
                 memory, memory_sequence_length,
                 trainable=True, name=None, **kwargs):
        super(AttentionRNNV1, self).__init__(name=name, trainable=trainable, **kwargs)
        attention_cell = AttentionWrapper(
            DecoderPreNetWrapper(GRUCell(num_units), prenets),
            BahdanauAttention(num_units, memory, memory_sequence_length),
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


class DecoderRNNV1(RNNCell):

    def __init__(self, out_units, attention_cell: AttentionRNNV1,
                 trainable=True, name=None, **kwargs):
        super(DecoderRNNV1, self).__init__(name=name, trainable=trainable, **kwargs)

        self._cell = MultiRNNCell([
            OutputProjectionWrapper(attention_cell, out_units),
            ResidualWrapper(GRUCell(out_units)),
            ResidualWrapper(GRUCell(out_units)),
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


class DecoderV1(tf.layers.Layer):

    def __init__(self, prenet_out_units=(256, 128), drop_rate=0.5,
                 attention_out_units=256,
                 decoder_out_units=256,
                 num_mels=80,
                 outputs_per_step=2,
                 max_iters=200,
                 trainable=True, name=None, **kwargs):
        super(DecoderV1, self).__init__(name=name, trainable=trainable, **kwargs)
        self._prenet_out_units = prenet_out_units
        self._drop_rate = drop_rate
        self.attention_out_units = attention_out_units
        self.decoder_out_units = decoder_out_units
        self.num_mels = num_mels
        self.outputs_per_step = outputs_per_step
        self.max_iters = max_iters

    def build(self, _):
        self.built = True

    def call(self, source, is_training=None, memory_sequence_length=None, target=None):
        assert is_training is not None

        prenets = tuple([PreNet(out_unit, is_training, self._drop_rate)
                         for out_unit in self._prenet_out_units])

        batch_size = tf.shape(source)[0]
        attention_cell = AttentionRNNV1(self.attention_out_units, prenets, source, memory_sequence_length)
        decoder_cell = DecoderRNNV1(self.decoder_out_units, attention_cell)
        output_cell = OutputProjectionWrapper(decoder_cell, self.num_mels * self.outputs_per_step)

        decoder_initial_state = output_cell.zero_state(batch_size, dtype=tf.float32)

        helper = TrainingHelper(target, self.num_mels,
                                self.outputs_per_step) if is_training else InferenceHelper(batch_size,
                                                                                           self.num_mels,
                                                                                           self.outputs_per_step)
        (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
            BasicDecoder(output_cell, helper, decoder_initial_state), maximum_iterations=self.max_iters)

        mel_output = tf.reshape(decoder_outputs, [batch_size, -1, self.num_mels])
        return mel_output, final_decoder_state
