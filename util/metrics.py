# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Visualizing metrics like alignments and spectrogram. """

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_alignment(alignments, text, _id, global_step, path):
    num_alignment = len(alignments)
    fig = plt.figure(figsize=(12, 16))
    for i, alignment in enumerate(alignments):
        ax = fig.add_subplot(num_alignment, 1, i + 1)
        im = ax.imshow(
            alignment,
            aspect='auto',
            origin='lower',
            interpolation='none')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Encoder timestep')
        ax.set_title("layer {}".format(i + 1))
    fig.subplots_adjust(wspace=0.4, hspace=0.6)
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}\ninput text: {str(text)}")
    fig.savefig(path, format='png')
    plt.close()


def plot_mel(mel, mel_predicted, text, _id, global_step, filename):
    from matplotlib import pylab as plt
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(2, 1, 1)
    im = ax.imshow(mel.T, origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(mel_predicted[:mel.shape[0], :].T,
                   origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}\ninput text: {str(text)}")
    fig.savefig(filename, format='png')
    plt.close()


def plot_spec(spec, spec_predicted, _id, global_step, filename):
    from matplotlib import pylab as plt
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(2, 1, 1)
    im = ax.imshow(spec.T, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(spec_predicted[:spec.shape[0], :].T,
                   origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}")
    fig.savefig(filename, format='png')
    plt.close()


def plot_predictions(alignments, mel, mel_predicted, spec, spec_predicted, text, _id, filename):
    fig = plt.figure(figsize=(12, 24))

    num_alignment = len(alignments)
    for i, alignment in enumerate(alignments):
        ax = fig.add_subplot(num_alignment + 4, 1, i + 1)
        im = ax.imshow(
            alignment,
            aspect='auto',
            origin='lower',
            interpolation='none')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Encoder timestep')
        ax.set_title("layer {}".format(i + 1))
    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    ax = fig.add_subplot(num_alignment + 4, 1, num_alignment + 1)
    im = ax.imshow(mel.T, origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(num_alignment + 4, 1, num_alignment + 2)
    im = ax.imshow(mel_predicted[:mel.shape[0], :].T,
                   origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)

    if spec is not None and spec_predicted is not None:
        ax = fig.add_subplot(num_alignment + 4, 1, num_alignment + 3)
        im = ax.imshow(spec.T, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax)
        ax = fig.add_subplot(num_alignment + 4, 1, num_alignment + 4)
        im = ax.imshow(spec_predicted[:spec.shape[0], :].T,
                       origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"record ID: {_id}\ninput text: {str(text)}")

    fig.savefig(filename, format='png')
    plt.close()
