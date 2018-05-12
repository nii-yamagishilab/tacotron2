import librosa
import numpy as np
import tensorflow as tf
import scipy
from functools import reduce
from hparams import hparams


def _build_mel_basis():
    n_fft = (hparams.num_freq - 1) * 2
    return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)


_mel_basis = _build_mel_basis()


def load_wav(path):
    return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_wav(wav, path):
    wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    scipy.io.wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))


def spectrogram(y):
    D = _stft(y)
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S)


def inv_spectrogram_tf(spectrogram):
    '''Builds computational graph to convert spectrogram to waveform using TensorFlow.
    '''
    S = _db_to_amp_tf(_denormalize_tf(spectrogram) + hparams.ref_level_db)
    return _griffin_lim_tf(tf.pow(S, hparams.power))


def melspectrogram(y):
    D = _stft(y)
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
    return _normalize(S)


def _griffin_lim_tf(S):
    '''TensorFlow implementation of Griffin-Lim
    Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
    '''
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    S = tf.expand_dims(S, axis=0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tf(S_complex)

    def reduce_func(y, i):
        est = _stft_tf(y)
        angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
        y = _istft_tf(S_complex * angles)
        return y

    y = reduce(reduce_func, range(hparams.griffin_lim_iters), y)
    return tf.squeeze(y, 0)


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _stft_tf(signals):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft)


def _istft_tf(stfts):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length


def _linear_to_mel(spectrogram):
    return np.dot(_mel_basis, spectrogram)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp_tf(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize_tf(S):
    return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
