# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Preprocess for LJSpeech dataset. """

from pyspark import SparkContext, RDD
import numpy as np
import os
import random
from util import tfrecord
from util.audio import Audio
from hparams import hparams
from datasets.corpus import Corpus, TargetMetaData, SourceMetaData, TextAndPath, target_metadata_to_tsv, \
    source_metadata_to_tsv, eos
from functools import reduce


class LJSpeech(Corpus):

    def __init__(self, in_dir, out_dir):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.audio = Audio(hparams)

    @property
    def record_ids(self):
        return map(lambda v: str(v), range(1, 13101))

    def record_file_path(self, record_id, kind):
        assert kind in ["source", "target"]
        return os.path.join(self.out_dir, f"ljspeech-{kind}-{int(record_id):05d}.tfrecord")

    @property
    def training_record_num(self):
        return 12000

    @property
    def validation_record_num(self):
        return 1000

    @property
    def test_record_num(self):
        return 100

    @property
    def training_source_files(self):
        with open(self.training_list_filepath, mode="r") as f:
            return [self.record_file_path(record_id, "source") for record_id in f]

    @property
    def training_target_files(self):
        with open(self.training_list_filepath, mode="r") as f:
            return [self.record_file_path(record_id, "target") for record_id in f]

    @property
    def validation_source_files(self):
        with open(self.validation_list_filepath, mode="r") as f:
            return [self.record_file_path(record_id, "source") for record_id in f]

    @property
    def validation_target_files(self):
        with open(self.validation_list_filepath, mode="r") as f:
            return [self.record_file_path(record_id, "target") for record_id in f]

    @property
    def test_source_files(self):
        with open(self.test_list_filepath, mode="r") as f:
            return [self.record_file_path(record_id, "source") for record_id in f]

    @property
    def test_target_files(self):
        with open(self.test_list_filepath, mode="r") as f:
            return [self.record_file_path(record_id, "target") for record_id in f]

    @property
    def training_list_filepath(self):
        return os.path.join(self.out_dir, "training_list.txt")

    @property
    def validation_list_filepath(self):
        return os.path.join(self.out_dir, "validation_list.txt")

    @property
    def test_list_filepath(self):
        return os.path.join(self.out_dir, "test_list.txt")

    def random_sample(self):
        ids = set(self.record_ids)
        validation_and_test = set(random.sample(ids, self.validation_record_num + self.test_record_num))
        test = set(random.sample(validation_and_test, self.test_record_num))
        validation = validation_and_test - test
        training = ids - validation_and_test
        return training, validation, test

    def text_and_path_rdd(self, sc: SparkContext):
        return sc.parallelize(
            self._extract_all_text_and_path())

    def process_targets(self, rdd: RDD):
        return rdd.mapValues(self._process_target)

    def process_sources(self, rdd: RDD):
        return rdd.mapValues(self._process_source)

    def aggregate_source_metadata(self, rdd: RDD):
        def map_fn(splitIndex, iterator):
            csv, max_len, count = reduce(
                lambda acc, kv: (
                    "\n".join([acc[0], source_metadata_to_tsv(kv[1])]), max(acc[1], len(kv[1].text)), acc[2] + 1),
                iterator, ("", 0, 0))
            filename = f"ljspeech-source-metadata-{splitIndex:03d}.tsv"
            filepath = os.path.join(self.out_dir, filename)
            with open(filepath, mode="w", encoding='utf-8') as f:
                f.write(csv)
            yield count, max_len

        return rdd.sortByKey().mapPartitionsWithIndex(
            map_fn, preservesPartitioning=True).fold(
            (0, 0), lambda acc, xy: (acc[0] + xy[0], max(acc[1], xy[1])))

    def aggregate_target_metadata(self, rdd: RDD):
        def map_fn(splitIndex, iterator):
            csv, max_len, count = reduce(
                lambda acc, kv: (
                    "\n".join([acc[0], target_metadata_to_tsv(kv[1])]), max(acc[1], kv[1].n_frames), acc[2] + 1),
                iterator, ("", 0, 0))
            filename = f"ljspeech-target-metadata-{splitIndex:03d}.tsv"
            filepath = os.path.join(self.out_dir, filename)
            with open(filepath, mode="w") as f:
                f.write(csv)
            yield count, max_len

        return rdd.sortByKey().mapPartitionsWithIndex(
            map_fn, preservesPartitioning=True).fold(
            (0, 0), lambda acc, xy: (acc[0] + xy[0], max(acc[1], xy[1])))

    def _extract_text_and_path(self, line, index):
        parts = line.strip().split('|')
        wav_path = os.path.join(self.in_dir, 'wavs', '%s.wav' % parts[0])
        text = parts[2]
        return TextAndPath(index, wav_path, None, text)

    def _extract_all_text_and_path(self):
        index = 1
        with open(os.path.join(self.in_dir, 'metadata.csv'), mode='r', encoding='utf-8') as f:
            for line in f:
                extracted = self._extract_text_and_path(line, index)
                if extracted is not None:
                    yield (index, extracted)
                    index += 1

    def _text_to_sequence(self, text):
        text = text.upper() if hparams.convert_to_upper else text
        sequence = [ord(c) for c in text] + [eos]
        sequence = np.array(sequence, dtype=np.int64)
        return sequence

    def _process_target(self, paths: TextAndPath):
        wav = self.audio.load_wav(paths.wav_path)
        spectrogram = self.audio.spectrogram(wav).astype(np.float32)
        n_frames = spectrogram.shape[1]
        mel_spectrogram = self.audio.melspectrogram(wav).astype(np.float32)
        filename = f"ljspeech-target-{paths.id:05d}.tfrecord"
        filepath = os.path.join(self.out_dir, filename)
        tfrecord.write_preprocessed_target_data(paths.id, spectrogram.T, mel_spectrogram.T, filepath)
        return TargetMetaData(paths.id, filepath, n_frames)

    def _process_source(self, paths: TextAndPath):
        sequence = self._text_to_sequence(paths.text)
        filename = f"ljspeech-source-{paths.id:05d}.tfrecord"
        filepath = os.path.join(self.out_dir, filename)
        tfrecord.write_preprocessed_source_data2(paths.id, paths.text, sequence, paths.text, sequence, filepath)
        return SourceMetaData(paths.id, filepath, paths.text)


def instantiate(in_dir, out_dir):
    return LJSpeech(in_dir, out_dir)
