# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================


"""Trainining script for seq2seq text-to-speech synthesis model.
Usage: train_postnet.py [options]

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
from datasets.dataset import PostNetDatasetSource
from tacotron.models import TacotronV1PostNetModel
from hparams import hparams, hparams_debug_string
from util.audio import Audio


def train_and_evaluate(hparams, model_dir, train_target_files, eval_target_files):
    audio = Audio(hparams)

    def train_input_fn():
        shuffled_train_target_files = list(train_target_files)
        shuffle(shuffled_train_target_files)
        target = tf.data.TFRecordDataset([t for t in shuffled_train_target_files])

        dataset = PostNetDatasetSource(target, hparams)
        batched = dataset.create_source_and_target().filter_by_max_output_length().repeat().shuffle(
            hparams.suffle_buffer_size).group_by_batch()
        return batched.dataset

    def eval_input_fn():
        shuffled_eval_target_files = list(eval_target_files)
        shuffle(shuffled_eval_target_files)
        target = tf.data.TFRecordDataset([t for t in shuffled_eval_target_files])

        dataset = PostNetDatasetSource(target, hparams)
        dataset = dataset.create_source_and_target().filter_by_max_output_length().repeat().group_by_batch(batch_size=1)
        return dataset.dataset

    run_config = tf.estimator.RunConfig(save_summary_steps=hparams.save_summary_steps,
                                        log_step_count_steps=hparams.log_step_count_steps)
    estimator = TacotronV1PostNetModel(hparams, audio, model_dir, config=run_config)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=hparams.num_evaluation_steps,
                                      throttle_secs=hparams.eval_throttle_secs,
                                      start_delay_secs=hparams.eval_start_delay_secs)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


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
                       corpus_instance.training_target_files,
                       corpus_instance.validation_target_files)


if __name__ == '__main__':
    main()
