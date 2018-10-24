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


"""
Preprocess dataset
usage: preprocess.py [options] <name> <in_dir> <out_dir>


options:
    --source-only            Process source only.
    --target-only            Process target only.
    -h, --help               Show help message.

"""

from docopt import docopt
import importlib
from pyspark import SparkContext

if __name__ == "__main__":
    args = docopt(__doc__)
    name = args["<name>"]
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    source_only = args["--source-only"]
    target_only = args["--target-only"]
    source_and_target = source_only is None and target_only is None

    assert name in ["blizzard2012", "ljspeech"]
    mod = importlib.import_module("datasets." + name)
    instance = mod.instantiate(in_dir, out_dir)

    sc = SparkContext()
    if target_only or source_and_target:
        target_metadata = instance.process_targets(
            instance.text_and_path_rdd(sc))
        target_num, max_target_len = instance.aggregate_target_metadata(target_metadata)
        print(f"number of target records: {target_num}, max target length: {max_target_len}")

    if source_only or source_and_target:
        source_meta = instance.process_sources(
            instance.text_and_path_rdd(sc))
        source_num, max_source_len = instance.aggregate_source_metadata(source_meta)
        print(f"number of source records: {source_num}, max source length: {max_source_len}")
