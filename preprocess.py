# coding: utf-8
"""
Preprocess dataset
usage: preprocess.py <name> <in_dir> <out_dir>

"""

from docopt import docopt
import importlib
from pyspark import SparkContext

if __name__ == "__main__":
    args = docopt(__doc__)
    name = args["<name>"]
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]

    assert name in ["blizzard2012"]
    mod = importlib.import_module("datasets." + name)
    instance = mod.instantiate(in_dir, out_dir)

    sc = SparkContext()
    target_metadata = instance.process_targets(
        instance.text_and_path_rdd(sc))
    target_num, max_target_len = instance.aggregate_target_metadata(target_metadata)
    print(f"number of target records: {target_num}, max target length: {max_target_len}")

    source_meta = instance.process_sources(
        instance.text_and_path_rdd(sc))
    source_num, max_source_len = instance.aggregate_source_metadata(source_meta)
    print(f"number of source records: {source_num}, max source length: {max_source_len}")