# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
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