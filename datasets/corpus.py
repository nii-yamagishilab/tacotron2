from abc import abstractmethod


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