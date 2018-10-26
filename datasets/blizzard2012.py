# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Preprocess for Blizzard 2012. """

from pyspark import SparkContext, RDD
import numpy as np
import os
from util import tfrecord
from util.audio import Audio
from hparams import hparams
from datasets.corpus import Corpus, TargetMetaData, SourceMetaData, TextAndPath, target_metadata_to_tsv, \
    source_metadata_to_tsv, eos
from functools import reduce


class Blizzard2012(Corpus):

    def __init__(self, in_dir, out_dir):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.books = [
            'ATrampAbroad',
            'TheManThatCorruptedHadleyburg',
            'LifeOnTheMississippi',
            'TheAdventuresOfTomSawyer',
        ]
        self._end_buffer = 0.05
        self._min_confidence = 90
        self.audio = Audio(hparams)

    @property
    def training_source_files(self):
        return [os.path.join(self.out_dir, f"blizzard2012-source-{record_id:05d}.tfrecord") for record_id in
                range(321, 23204)]

    @property
    def training_target_files(self):
        return [os.path.join(self.out_dir, f"blizzard2012-target-{record_id:05d}.tfrecord") for record_id in
                range(321, 23204)]

    @property
    def validation_source_files(self):
        return [os.path.join(self.out_dir, f"blizzard2012-source-{record_id:05d}.tfrecord") for record_id in
                range(11, 321)]

    @property
    def validation_target_files(self):
        return [os.path.join(self.out_dir, f"blizzard2012-target-{record_id:05d}.tfrecord") for record_id in
                range(11, 321)]

    @property
    def test_source_files(self):
        return [os.path.join(self.out_dir, f"blizzard2012-source-{record_id:05d}.tfrecord") for record_id in
                range(1, 11)]

    @property
    def test_target_files(self):
        return [os.path.join(self.out_dir, f"blizzard2012-target-{record_id:05d}.tfrecord") for record_id in
                range(1, 11)]

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
            filename = f"blizzard2012-source-metadata-{splitIndex:03d}.tsv"
            filepath = os.path.join(self.out_dir, filename)
            with open(filepath, mode="w") as f:
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
            filename = f"blizzard2012-target-metadata-{splitIndex:03d}.tsv"
            filepath = os.path.join(self.out_dir, filename)
            with open(filepath, mode="w") as f:
                f.write(csv)
            yield count, max_len

        return rdd.sortByKey().mapPartitionsWithIndex(
            map_fn, preservesPartitioning=True).fold(
            (0, 0), lambda acc, xy: (acc[0] + xy[0], max(acc[1], xy[1])))

    def _extract_text_and_path(self, book, line, index):
        parts = line.strip().split('\t')
        if line[0] is not '#' and len(parts) == 8 and float(parts[3]) > self._min_confidence:
            wav_path = os.path.join(self.in_dir, book, 'wav', '%s.wav' % parts[0])
            labels_path = os.path.join(self.in_dir, book, 'lab', '%s.lab' % parts[0])
            text = parts[5]
            return TextAndPath(index, wav_path, labels_path, text)

    def _extract_all_text_and_path(self):
        index = 1
        for book in self.books:
            with open(os.path.join(self.in_dir, book, 'sentence_index.txt'), mode='r') as f:
                for line in f:
                    extracted = self._extract_text_and_path(book, line, index)
                    if extracted is not None:
                        yield (index, extracted)
                        index += 1

    def _load_labels(self, path):
        labels = []
        with open(os.path.join(path)) as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 3:
                    labels.append((float(parts[0]), ' '.join(parts[2:])))
        start = 0
        end = None
        if labels[0][1] == 'sil':
            start = labels[0][0]
        if labels[-1][1] == 'sil':
            end = labels[-2][0] + self._end_buffer
        return (start, end)

    def _text_to_sequence(self, text):
        sequence = [ord(c) for c in text] + [eos]
        sequence = np.array(sequence, dtype=np.int64)
        return sequence

    def _process_target(self, paths: TextAndPath):
        wav = self.audio.load_wav(paths.wav_path)
        start_offset, end_offset = self._load_labels(paths.labels_path)
        start = int(start_offset * hparams.sample_rate)
        end = int(end_offset * hparams.sample_rate) if end_offset is not None else -1
        wav = wav[start:end]
        spectrogram = self.audio.spectrogram(wav).astype(np.float32)
        n_frames = spectrogram.shape[1]
        mel_spectrogram = self.audio.melspectrogram(wav).astype(np.float32)
        filename = f"blizzard2012-target-{paths.id:05d}.tfrecord"
        filepath = os.path.join(self.out_dir, filename)
        tfrecord.write_preprocessed_target_data(paths.id, spectrogram.T, mel_spectrogram.T, filepath)
        return TargetMetaData(paths.id, filepath, n_frames)

    def _process_source(self, paths: TextAndPath):
        sequence = self._text_to_sequence(paths.text)
        filename = f"blizzard2012-source-{paths.id:05d}.tfrecord"
        filepath = os.path.join(self.out_dir, filename)
        tfrecord.write_preprocessed_source_data2(paths.id, paths.text, sequence, paths.text, sequence, filepath)
        return SourceMetaData(paths.id, filepath, paths.text)


def instantiate(in_dir, out_dir):
    return Blizzard2012(in_dir, out_dir)
