from abc import abstractmethod
from collections import namedtuple


class Corpus:

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