# BSD 3-Clause License
#
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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