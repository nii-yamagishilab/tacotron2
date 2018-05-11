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
from datasets.dataset import DatasetSource
from tacotron.models import SingleSpeakerTacotronV1Model
from hparams import hparams, hparams_debug_string


def train_and_evaluate(hparams, model_dir, train_source_files, train_target_files, eval_source_files,
                       eval_target_files):
    def train_input_fn():
        source = tf.data.TFRecordDataset(list(train_source_files))
        target = tf.data.TFRecordDataset(list(train_target_files))

        dataset = DatasetSource(source, target, hparams)
        batched = dataset.prepare_and_zip().filter_by_max_output_length().repeat().shuffle(
            hparams.batch_size).group_by_batch()
        return batched.dataset

    def eval_input_fn():
        source = tf.data.TFRecordDataset(list(eval_source_files))
        target = tf.data.TFRecordDataset(list(eval_target_files))

        dataset = DatasetSource(source, target, hparams)
        dataset = dataset.prepare_and_zip().filter_by_max_output_length().repeat().group_by_batch(batch_size=1)
        return dataset.dataset

    run_config = tf.estimator.RunConfig(save_summary_steps=hparams.save_summary_steps,
                                        log_step_count_steps=hparams.log_step_count_steps)
    estimator = SingleSpeakerTacotronV1Model(hparams, model_dir, config=run_config)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    data_root = args["--data-root"]
    dataset_name = args["--dataset"]
    assert dataset_name in ["blizzard2012"]
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
