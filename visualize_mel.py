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
usage: visualize_mel.py [options] <filename>

options:
    --output-prefix=<prefix>        output filename prefix

"""
from docopt import docopt
import numpy as np
import tensorflow as tf
from collections import namedtuple
import matplotlib
import os

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class TrainingResult(
    namedtuple("TrainingResult",
               ["global_step", "id", "text", "predicted_mel", "ground_truth_mel", "alignments"])):
    pass


def read_training_result(filename):
    record_iterator = tf.python_io.tf_record_iterator(filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        global_step = example.features.feature['global_step'].int64_list.value[0]
        batch_size = example.features.feature['batch_size'].int64_list.value[0]
        id = example.features.feature['id'].int64_list.value
        text = example.features.feature['text'].bytes_list.value
        predicted_mel = example.features.feature['predicted_mel'].bytes_list.value
        ground_truth_mel = example.features.feature['ground_truth_mel'].bytes_list.value
        mel_length = example.features.feature['mel_length'].int64_list.value
        mel_width = example.features.feature['mel_width'].int64_list.value[0]
        alignment = example.features.feature['alignment'].bytes_list.value
        alignment_source_length = example.features.feature['alignment_source_length'].int64_list.value
        alignment_target_length = example.features.feature['alignment_target_length'].int64_list.value

        texts = (t.decode('utf-8') for t in text)
        alignments = [np.frombuffer(align, dtype=np.float32).reshape([batch_size, src_len, tgt_len]) for
                      align, src_len, tgt_len in
                      zip(alignment, alignment_source_length, alignment_target_length)]
        alignments = [[a[i].T for a in alignments] for i in range(batch_size)]
        predicted_mels = (np.frombuffer(mel, dtype=np.float32).reshape([-1, mel_width]) for mel, mel_len in
                          zip(predicted_mel, mel_length))
        ground_truth_mels = (np.frombuffer(mel, dtype=np.float32).reshape([mel_len, mel_width]) for mel, mel_len in
                             zip(ground_truth_mel, mel_length))

        for id, text, align, pred_mel, gt_mel in zip(id, texts, alignments, predicted_mels, ground_truth_mels):
            yield TrainingResult(
                global_step=global_step,
                id=id,
                text=text,
                predicted_mel=pred_mel,
                ground_truth_mel=gt_mel,
                alignments=align,
            )


def plot_mel(mel, mel_predicted, record_id, text, filename):
    from matplotlib import pylab as plt
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(2, 1, 1)
    im = ax.imshow(mel.T, origin="lower bottom", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(mel_predicted[:mel.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"record ID: {record_id}, input text: \"{str(text)}\"")
    fig.savefig(filename, format='png')
    plt.close()


if __name__ == "__main__":
    args = docopt(__doc__)
    filename = args["<filename>"]
    prefix = args["--output-prefix"] or "mel_"
    output_base_filename, _ = os.path.splitext(os.path.basename(filename))
    output_dir = os.path.dirname(filename)
    output_filename = prefix + output_base_filename + "_{}.png"

    for result in read_training_result(filename):
        plot_mel(result.ground_truth_mel,
                 result.predicted_mel,
                 result.id,
                 result.text,
                 os.path.join(output_dir, output_filename).format(result.id))
