# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================


"""Synthesis script for seq2seq text-to-speech synthesis model.
Usage: synthesize.py [options]

Options:
    --data-root=<dir>               Directory contains preprocessed features.
    --checkpoint-dir=<dir>          Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>              Hyper parameters. [default: ].
    --dataset=<name>                Dataset name.
    --postnet-checkpoint-dir=<path> Restore postnet model from checkpoint path if given.
    -h, --help                      Show this help message and exit
"""

from docopt import docopt
import tensorflow as tf
import importlib
import os
from datasets.dataset import DatasetSource, PostNetDatasetSource, PredictedMel
from tacotron.models import SingleSpeakerTacotronV1Model, TacotronV1PostNetModel
from hparams import hparams, hparams_debug_string
from util.audio import Audio


def predict(hparams,
            model_dir, postnet_model_dir,
            test_source_files, test_target_files):
    audio = Audio(hparams)

    def predict_input_fn():
        source = tf.data.TFRecordDataset(list(test_source_files))
        target = tf.data.TFRecordDataset(list(test_target_files))
        dataset = DatasetSource(source, target, hparams)
        batched = dataset.prepare_and_zip().filter_by_max_output_length().group_by_batch(batch_size=1)
        return batched.dataset

    estimator = SingleSpeakerTacotronV1Model(hparams, model_dir)

    predictions = map(
        lambda p: PredictedMel(p["id"], p["mel"], p["mel"].shape[1], p["mel"].shape[0], p["alignment"], p["source"],
                               p["text"]),
        estimator.predict(predict_input_fn))

    def predict_postnet_input_fn():
        prediction_dataset = tf.data.Dataset.from_generator(lambda: predictions,
                                                            output_types=PredictedMel(tf.int64,
                                                                                      tf.float32,
                                                                                      tf.int64,
                                                                                      tf.int64,
                                                                                      tf.float32,
                                                                                      tf.int64,
                                                                                      tf.string))
        target = tf.data.TFRecordDataset(list(test_target_files))
        dataset = PostNetDatasetSource(target, hparams)
        batched = dataset.create_source_and_target().filter_by_max_output_length().combine_with_prediction(
            prediction_dataset).expand_batch_dim()
        return batched.dataset

    postnet_estimator = TacotronV1PostNetModel(hparams, audio, postnet_model_dir)

    for v in postnet_estimator.predict(predict_postnet_input_fn):
        filename = f"{v['id']}.wav"
        filepath = os.path.join(postnet_model_dir, filename)
        audio.save_wav(v["audio"], filepath)


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    postnet_checkpoint_dir = args["--postnet-checkpoint-dir"]
    data_root = args["--data-root"]
    dataset_name = args["--dataset"]
    assert dataset_name in ["blizzard2012"]
    corpus = importlib.import_module("datasets." + dataset_name)
    corpus_instance = corpus.instantiate(in_dir="", out_dir=data_root)

    hparams.parse(args["--hparams"])
    print(hparams_debug_string())

    tf.logging.set_verbosity(tf.logging.INFO)
    predict(hparams,
            checkpoint_dir,
            postnet_checkpoint_dir,
            corpus_instance.test_source_files,
            corpus_instance.test_target_files, )


if __name__ == '__main__':
    main()
