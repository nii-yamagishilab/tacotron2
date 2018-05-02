import librosa
import numpy as np
from hparams import hparams


def _build_mel_basis():
    n_fft = (hparams.num_freq - 1) * 2
    return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)


_mel_basis = _build_mel_basis()


def load_wav(path):
    return librosa.core.load(path, sr=hparams.sample_rate)[0]


def spectrogram(y):
    D = _stft(y)
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S)


def melspectrogram(y):
    D = _stft(y)
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
    return _normalize(S)


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length


def _linear_to_mel(spectrogram):
    return np.dot(_mel_basis, spectrogram)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)
