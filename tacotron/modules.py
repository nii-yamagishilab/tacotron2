import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from functools import reduce


class Embedding(tf.layers.Layer):

    def __init__(self, num_symbols, embedding_dim,
                 trainable=True, name=None, **kwargs):
        super(Embedding, self).__init__(name=name, trainable=trainable, **kwargs)
        self._num_symbols = num_symbols
        self._embedding_dim = embedding_dim

    def build(self, _):
        self._embedding = self.add_variable("embedding", shape=[self._num_symbols, self._embedding_dim],
                                            dtype=tf.float32)
        self.built = True

    def call(self, inputs, **kwargs):
        return tf.nn.embedding_lookup(self._embedding, inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self._embedding_dim])


class PreNet(tf.layers.Layer):

    def __init__(self, out_units, is_training, drop_rate=0.5,
                 trainable=True, name=None, **kwargs):
        super(PreNet, self).__init__(name=name, trainable=trainable, **kwargs)
        self.out_units = out_units
        self.drop_rate = drop_rate
        self.is_training = is_training
        self.dense = tf.layers.Dense(out_units, activation=tf.nn.relu)

    def build(self, _):
        self.built = True

    def call(self, inputs, **kwargs):
        dense = self.dense(inputs)
        dropout = tf.layers.dropout(dense, rate=self.drop_rate, training=self.is_training)
        return dropout

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)


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
                 trainable=True, name=None, **kwargs):
        super(Conv1d, self).__init__(name=name, trainable=trainable, **kwargs)
        self.is_training = is_training
        self.conv1d = tf.layers.Conv1D(out_channels, kernel_size, activation=activation, padding="SAME")

    def build(self, _):
        self.built = True

    def call(self, inputs, **kwargs):
        conv1d = self.conv1d(inputs)
        batch_normalization = tf.layers.batch_normalization(conv1d, training=self.is_training)
        return batch_normalization

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
