# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Base class for corpora. """



from abc import abstractmethod
from collections import namedtuple


class Corpus:

    @property
    def record_ids(self):
        raise NotImplementedError("record_ids")

    @abstractmethod
    def random_sample(self):
        raise NotImplementedError("random_sample")

    @property
    @abstractmethod
    def training_source_files(self):
        raise NotImplementedError("training_source_files")

    @property
    @abstractmethod
    def training_target_files(self):
        raise NotImplementedError("training_target_files")

    @property
    @abstractmethod
    def validation_source_files(self):
        raise NotImplementedError("test_source_files")

    @property
    @abstractmethod
    def validation_target_files(self):
        raise NotImplementedError("test_target_files")

    @property
    @abstractmethod
    def training_list_filepath(self):
        raise NotImplementedError("training_list_filepath")

    @property
    @abstractmethod
    def validation_list_filepath(self):
        raise NotImplementedError("validation_list_filepath")

    @property
    @abstractmethod
    def test_list_filepath(self):
        raise NotImplementedError("test_list_filepath")


class TextAndPath(namedtuple("TextAndPath", ["id", "wav_path", "labels_path", "text"])):
    pass


class SourceMetaData(namedtuple("SourceMetaData", ["id", "filepath", "text"])):
    pass


def source_metadata_to_tsv(meta):
    return "\t".join([str(meta.id), meta.filepath, meta.text])


class TargetMetaData(namedtuple("TargetMetaData", ["id", "filepath", "n_frames"])):
    pass


def target_metadata_to_tsv(meta):
    return "\t".join([str(meta.id), meta.filepath, str(meta.n_frames)])


eos = 1