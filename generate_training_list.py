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


"""Generate record ID lists for training, validation, test.
Usage: generate_training_list.py [options]

Options:
    --data-root=<dir>            Directory contains preprocessed features.
    --dataset=<name>             Dataset name.

"""

import importlib
from docopt import docopt
from datasets.corpus import Corpus

if __name__ == "__main__":
    args = docopt(__doc__)
    data_root = args["--data-root"]
    dataset_name = args["--dataset"]

    assert dataset_name in ["ljspeech"]

    corpus = importlib.import_module("datasets." + dataset_name)
    corpus_instance: Corpus = corpus.instantiate(in_dir="", out_dir=data_root)

    training, validation, test = corpus_instance.random_sample()

    with open(corpus_instance.training_list_filepath, mode="w") as f:
        f.write("\n".join(training))
        print("Generated " + corpus_instance.training_list_filepath)

    with open(corpus_instance.validation_list_filepath, mode="w") as f:
        f.write("\n".join(validation))
        print("Generated " + corpus_instance.validation_list_filepath)

    with open(corpus_instance.test_list_filepath, mode="w") as f:
        f.write("\n".join(test))
        print("Generated " + corpus_instance.test_list_filepath)