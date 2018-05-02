from pyspark import SparkContext, RDD
import numpy as np
import os
from collections import namedtuple
from util import audio, tfrecord
from hparams import hparams


class TextAndPath(namedtuple("TextAndPath", ["id", "wav_path", "labels_path", "text"])):
    pass


_eos = 1


class Blizzard2012:

    def __init__(self, in_dir, out_dir):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.books = [
            'ATrampAbroad',
            'TheManThatCorruptedHadleyburg',
            'LifeOnTheMississippi',
            'TheAdventuresOfTomSawyer',
        ]
        self._max_out_length = 700
        self._end_buffer = 0.05
        self._min_confidence = 90

    def text_and_path_rdd(self, sc: SparkContext):
        return sc.parallelize(self._extract_all_text_and_path())

    def process_targets(self, rdd: RDD):
        return rdd.map(self._process_target, preservesPartitioning=True)

    def process_sources(self, rdd: RDD):
        return rdd.map(self._process_source, preservesPartitioning=True)

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
                        yield extracted
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
        sequence = [ord(c) for c in text] + [_eos]
        sequence = np.array(sequence, dtype=np.int64)
        return sequence

    def _process_target(self, paths: TextAndPath):
        wav = audio.load_wav(paths.wav_path)
        start_offset, end_offset = self._load_labels(paths.labels_path)
        start = int(start_offset * hparams.sample_rate)
        end = int(end_offset * hparams.sample_rate) if end_offset is not None else -1
        wav = wav[start:end]
        max_samples = self._max_out_length * hparams.frame_shift_ms / 1000 * hparams.sample_rate
        if len(wav) > max_samples:
            return None
        spectrogram = audio.spectrogram(wav).astype(np.float32)
        n_frames = spectrogram.shape[1]
        mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
        filename = f"blizzard2012-target-{paths.id:05d}.tfrecord".format(paths.id)
        filepath = os.path.join(self.out_dir, filename)
        tfrecord.write_preprocessed_target_data(paths.id, spectrogram.T, mel_spectrogram.T, filepath)
        return filename, n_frames

    def _process_source(self, paths: TextAndPath):
        sequence = self._text_to_sequence(paths.text)
        filename = f"blizzard2012-source-{paths.id:05d}.tfrecord".format(paths.id)
        filepath = os.path.join(self.out_dir, filename)
        tfrecord.write_preprocessed_source_data2(paths.id, paths.text, sequence, paths.text, sequence, filepath)


def instantiate(in_dir, out_dir):
    return Blizzard2012(in_dir, out_dir)
