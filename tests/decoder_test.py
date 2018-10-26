# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Tests about decoder at training and inference time. """

import tensorflow as tf
import numpy as np
from hypothesis import given, settings, unlimited, assume, HealthCheck
from hypothesis.strategies import integers, floats, composite
from hypothesis.extra.numpy import arrays
from tacotron.tacotron_v1 import DecoderV1

even_number = lambda x: x % 2 == 0


@composite
def memory_tensor(draw, batch_size, source_length=integers(5, 20),
                  embed_dim=integers(4, 20).filter(even_number), elements=floats(-1.0, 1.0)):
    il = draw(source_length)
    source_length = np.repeat(il, batch_size)
    md = draw(embed_dim)
    memory = draw(arrays(dtype=np.float32, shape=[batch_size, il, md], elements=elements))
    return memory, source_length


@composite
def all_args(draw, batch_size=integers(1, 3),
             num_mels=integers(2, 20), r=integers(1, 4)):
    bs = draw(batch_size)
    _in_dim = draw(num_mels)
    _r = draw(r)
    memory, source_length = draw(memory_tensor(bs))
    return memory, source_length, _in_dim, _r


class DecoderTest(tf.test.TestCase):

    @given(args=all_args())
    @settings(max_examples=10, timeout=unlimited, suppress_health_check=[HealthCheck.too_slow])
    def test_decoder(self, args):
        memory, source_length, num_mels, r = args

        max_iters = 10

        decoder = DecoderV1(drop_rate=0.0, max_iters=max_iters, num_mels=num_mels,
                            outputs_per_step=r)

        output_inference, stop_token_inference, alignment_inference = decoder(memory, is_training=False,
                                                                              memory_sequence_length=source_length)

        output_training, stop_token_training, alignment_training = decoder(memory, is_training=True,
                                                                           memory_sequence_length=source_length,
                                                                           target=output_inference)

        alignments_inference = tf.transpose(alignment_inference[0].alignment_history.stack(), [1, 2, 0])
        alignments_training = tf.transpose(alignment_training[0].alignment_history.stack(), [1, 2, 0])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            alignment_inference, alignment_training = sess.run([alignments_inference, alignments_training])
            self.assertAllClose(alignment_inference, alignment_training)
            output_inference, output_training = sess.run([output_inference, output_training])
            self.assertAllClose(output_inference, output_training)
            stop_token_inference, stop_token_training = sess.run([stop_token_inference, stop_token_training])
            self.assertAllClose(stop_token_inference, stop_token_training)
