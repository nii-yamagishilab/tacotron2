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
        ax.hlines(len(text), xmin=0, xmax=alignment.shape[1], colors=['red'])
    fig.subplots_adjust(wspace=0.4, hspace=0.6)
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}\ninput text: {str(text)}")
    fig.savefig(path, format='png')
    plt.close()


def plot_mel(mel, mel_predicted, text, _id, global_step, filename):
    from matplotlib import pylab as plt
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(2, 1, 1)
    im = ax.imshow(mel.T, origin="lower bottom", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(mel_predicted[:mel.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}\ninput text: {str(text)}")
    fig.savefig(filename, format='png')
    plt.close()
