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
""" Helpers. """

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
        return (
            tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim * self.n_feed_frame))

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
        return (
            tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim * self.n_feed_frame))

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
        termination_criteria = tf.greater(tf.nn.sigmoid(done), 0.5)
        minimum_requirement = tf.greater(time, self.min_iters)
        termination = tf.logical_and(termination_criteria, minimum_requirement)
        return tf.reduce_all(termination, axis=0)


class ValidationHelper(Helper):

    def __init__(self, targets, batch_size, output_dim, r, n_feed_frame=1, teacher_forcing=False):
        assert n_feed_frame <= r
        self._batch_size = batch_size
        self._output_dim = output_dim
        self._end_token = tf.tile([0.0], [output_dim * r])
        self.n_feed_frame = n_feed_frame
        self.num_steps = tf.shape(targets)[1] // r
        self.teacher_forcing = teacher_forcing
        self._targets = tf.reshape(targets,
                                   shape=tf.stack([self.batch_size, self.num_steps, tf.to_int32(output_dim * r)]))

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
        return (
            tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim * self.n_feed_frame))

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        output, done = outputs
        finished = (time + 1 >= self.num_steps)
        next_inputs = self._targets[:, time,
                      -self._output_dim * self.n_feed_frame:] if self.teacher_forcing else output[:,
                                                                                           -self._output_dim * self.n_feed_frame:]
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
        return (
            tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim * self.n_feed_frame))

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
