import tensorflow as tf
import numpy as np
from collections import namedtuple
from collections.abc import Iterable


class PreprocessedSourceData(namedtuple("PreprocessedSourceData",
                                        ["id", "text", "source", "source_length", "text2", "source2",
                                         "source_length2"])):
    pass


class PreprocessedTargetData(namedtuple("PreprocessedTargetData",
                                        ["id", "spec", "spec_width", "mel", "mel_width", "target_length"])):
    pass


def bytes_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tfrecord(example: tf.train.Example, filename: str):
    with tf.python_io.TFRecordWriter(filename) as writer:
        writer.write(example.SerializeToString())


def write_preprocessed_target_data(id: int, spec: np.ndarray, mel: np.ndarray, filename: str):
    raw_spec = spec.tostring()
    raw_mel = mel.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([id]),
        'spec': bytes_feature([raw_spec]),
        'spec_width': int64_feature([spec.shape[1]]),
        'mel': bytes_feature([raw_mel]),
        'mel_width': int64_feature([mel.shape[1]]),
        'target_length': int64_feature([len(mel)]),
    }))
    write_tfrecord(example, filename)


def write_preprocessed_source_data2(id: int, text1: str, source1: np.ndarray, text2: str, source2: np.ndarray,
                                    filename: str):
    raw_source1 = source1.tostring()
    raw_source2 = source2.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([id]),
        'text': bytes_feature([text1.encode('utf-8'), text2.encode('utf-8')]),
        'source': bytes_feature([raw_source1, raw_source2]),
        'source_length': int64_feature([len(source1), len(source2)]),
    }))
    write_tfrecord(example, filename)
