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


"""Trainining script for seq2seq text-to-speech synthesis model.
Usage: train.py [options]

Options:
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters. [default: ].
    --dataset=<name>             Dataset name.
    --checkpoint=<path>          Restore model from checkpoint path if given.
    -h, --help                   Show this help message and exit
"""

from docopt import docopt
import tensorflow as tf
import importlib
from random import shuffle
from multiprocessing import cpu_count
from datasets.dataset import DatasetSource
from tacotron.models import SingleSpeakerTacotronV1Model
from hparams import hparams, hparams_debug_string


def train_and_evaluate(hparams, model_dir, train_source_files, train_target_files, eval_source_files,
                       eval_target_files):
    interleave_parallelism = get_parallelism(hparams.interleave_cycle_length_cpu_factor,
                                             hparams.interleave_cycle_length_min,
                                             hparams.interleave_cycle_length_max)

    tf.logging.info("Interleave parallelism is %d.", interleave_parallelism)

    def train_input_fn():
        source_and_target_files = list(zip(train_source_files, train_target_files))
        shuffle(source_and_target_files)
        source = (s for s, _ in source_and_target_files)
        target = (t for _, t in source_and_target_files)

        dataset = DatasetSource.create_from_tfrecord_files(source, target, hparams,
                                                           cycle_length=interleave_parallelism,
                                                           buffer_output_elements=hparams.interleave_buffer_output_elements,
                                                           prefetch_input_elements=hparams.interleave_prefetch_input_elements)
        batched = dataset.prepare_and_zip().filter_by_max_output_length().shuffle_and_repeat(
            hparams.suffle_buffer_size).group_by_batch().prefetch(hparams.prefetch_buffer_size)
        return batched.dataset

    def eval_input_fn():
        source_and_target_files = list(zip(eval_source_files, eval_target_files))
        shuffle(source_and_target_files)
        source = tf.data.TFRecordDataset([s for s, _ in source_and_target_files])
        target = tf.data.TFRecordDataset([t for _, t in source_and_target_files])

        dataset = DatasetSource(source, target, hparams)
        dataset = dataset.prepare_and_zip().filter_by_max_output_length().repeat().group_by_batch(batch_size=1)
        return dataset.dataset

    run_config = tf.estimator.RunConfig(save_summary_steps=hparams.save_summary_steps,
                                        log_step_count_steps=hparams.log_step_count_steps)
    estimator = SingleSpeakerTacotronV1Model(hparams, model_dir, config=run_config)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=hparams.num_evaluation_steps,
                                      throttle_secs=hparams.eval_throttle_secs,
                                      start_delay_secs=hparams.eval_start_delay_secs)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def get_parallelism(factor, min_value, max_value):
    return min(max(int(cpu_count() * factor), min_value), max_value)


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    data_root = args["--data-root"]
    dataset_name = args["--dataset"]
    assert dataset_name in ["blizzard2012", "ljspeech"]
    corpus = importlib.import_module("datasets." + dataset_name)
    corpus_instance = corpus.instantiate(in_dir="", out_dir=data_root)

    hparams.parse(args["--hparams"])
    print(hparams_debug_string())

    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_evaluate(hparams,
                       checkpoint_dir,
                       corpus_instance.training_source_files,
                       corpus_instance.training_target_files,
                       corpus_instance.validation_source_files,
                       corpus_instance.validation_target_files)


if __name__ == '__main__':
    main()
