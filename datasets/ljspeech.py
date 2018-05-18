from pyspark import SparkContext, RDD
import numpy as np
import os
from collections import namedtuple
from util import audio, tfrecord
from hparams import hparams
from datasets.corpus import Corpus, TargetMetaData, SourceMetaData, TextAndPath, target_metadata_to_tsv, \
    source_metadata_to_tsv, eos
from functools import reduce


class LJSpeech(Corpus):

    def __init__(self, in_dir, out_dir):
        self.in_dir = in_dir
        self.out_dir = out_dir

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
        wav = audio.load_wav(paths.wav_path)
        spectrogram = audio.spectrogram(wav).astype(np.float32)
        n_frames = spectrogram.shape[1]
        mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
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
