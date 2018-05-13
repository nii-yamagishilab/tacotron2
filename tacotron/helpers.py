import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper


class InferenceHelper(Helper):

    def __init__(self, batch_size, output_dim, r, n_feed_frame=1):
        assert n_feed_frame <= r
        self._batch_size = batch_size
        self._output_dim = output_dim
        self._end_token = tf.tile([0.0], [output_dim * r])
        self.n_feed_frame = n_feed_frame

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        outputs, done = outputs
        finished = tf.reduce_all(tf.equal(outputs, self._end_token), axis=1)
        next_inputs = outputs[:, -self._output_dim * self.n_feed_frame:]
        next_inputs.set_shape([outputs.get_shape()[0].value, self._output_dim * self.n_feed_frame])
        return (finished, next_inputs, state)


class StopTokenBasedInferenceHelper(Helper):

    def __init__(self, batch_size, output_dim, r, n_feed_frame=1, min_iters=10):
        assert n_feed_frame <= r
        self._batch_size = batch_size
        self._output_dim = output_dim
        self.n_feed_frame = n_feed_frame
        self.min_iters = min_iters

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        output, done = outputs
        finished = self.is_finished(done, time)
        next_inputs = output[:, -self._output_dim * self.n_feed_frame:]
        next_inputs.set_shape([output.get_shape()[0].value, self._output_dim * self.n_feed_frame])
        return (finished, next_inputs, state)

    def is_finished(self, done, time):
        termination_criteria = tf.greater(done, 0.5)
        minimum_requirement = tf.greater(time, self.min_iters)
        termination = tf.logical_and(termination_criteria, minimum_requirement)
        return tf.reduce_all(termination, axis=0)


class ValidationHelper(Helper):

    def __init__(self, targets, batch_size, output_dim, r, n_feed_frame=1):
        assert n_feed_frame <= r
        self._batch_size = batch_size
        self._output_dim = output_dim
        self._end_token = tf.tile([0.0], [output_dim * r])
        self.n_feed_frame = n_feed_frame
        self.num_steps = tf.shape(targets)[1] // r

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        output, done = outputs
        finished = (time + 1 >= self.num_steps)
        next_inputs = output[:, -self._output_dim * self.n_feed_frame:]
        next_inputs.set_shape([output.get_shape()[0].value, self._output_dim * self.n_feed_frame])
        return (finished, next_inputs, state)


class TrainingHelper(Helper):

    def __init__(self, targets, output_dim, r, n_feed_frame=1):
        assert n_feed_frame <= r
        t_shape = tf.shape(targets)
        self._batch_size = t_shape[0]
        self._output_dim = output_dim
        self.n_feed_frame = n_feed_frame

        self._targets = tf.reshape(targets,
                                   shape=tf.stack([self.batch_size, t_shape[1] // r, tf.to_int32(output_dim * r)]))
        self._targets.set_shape((targets.get_shape()[0].value, None, output_dim * r))

        # Use full length for every target because we don't want to mask the padding frames
        num_steps = tf.shape(self._targets)[1]
        self._lengths = tf.tile([num_steps], [self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        output, done = outputs
        finished = (time + 1 >= self._lengths)
        next_inputs = self._targets[:, time, -self._output_dim * self.n_feed_frame:]
        next_inputs.set_shape([output.get_shape()[0].value, self._output_dim * self.n_feed_frame])
        return (finished, next_inputs, state)


def _go_frames(batch_size, output_dim, n_feed_frame=1):
    return tf.tile([[0.0]], [batch_size, output_dim * n_feed_frame])