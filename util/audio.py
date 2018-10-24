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
# Copyright (c) 2017 Keith Ito
#               2018 Modified by Yusuke Yasuda
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
""" Audio utilities.
Modified from keithito's implementation.
https://github.com/keithito/tacotron/blob/master/util/audio.py
"""

import librosa
import numpy as np
import tensorflow as tf
import scipy
from functools import reduce


class Audio:
    def __init__(self, hparams):
        self.hparams = hparams
        self._mel_basis = self._build_mel_basis()

    def _build_mel_basis(self):
        n_fft = (self.hparams.num_freq - 1) * 2
        return librosa.filters.mel(self.hparams.sample_rate, n_fft, n_mels=self.hparams.num_mels)

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.hparams.sample_rate)[0]

    def save_wav(self, wav, path):
        wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
        scipy.io.wavfile.write(path, self.hparams.sample_rate, wav.astype(np.int16))

    def spectrogram(self, y):
        D = self._stft(y)
        S = self._amp_to_db(np.abs(D)) - self.hparams.ref_level_db
        return self._normalize(S)

    def inv_amp_tf(self, spectrogram):
        S = self._db_to_amp_tf(self._denormalize_tf(spectrogram) + self.hparams.ref_level_db)
        return S

    def inv_spectrogram_tf(self, spectrogram):
        '''Builds computational graph to convert spectrogram to waveform using TensorFlow.
        '''
        S = self.inv_amp_tf(spectrogram)
        return self._griffin_lim_tf(tf.pow(S, self.hparams.power))

    def melspectrogram(self, y):
        D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.hparams.ref_level_db
        return self._normalize(S)

    def _griffin_lim_tf(self, S):
        '''TensorFlow implementation of Griffin-Lim
        Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
        '''
        # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
        S = tf.expand_dims(S, axis=0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = self._istft_tf(S_complex)

        def reduce_func(y, i):
            est = self._stft_tf(y)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = self._istft_tf(S_complex * angles)
            return y

        y = reduce(reduce_func, range(self.hparams.griffin_lim_iters), y)
        return tf.squeeze(y, 0)

    def _stft(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def _stft_tf(self, signals):
        n_fft, hop_length, win_length = self._stft_parameters()
        return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft)

    def _istft_tf(self, stfts):
        n_fft, hop_length, win_length = self._stft_parameters()
        return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)

    def _stft_parameters(self):
        n_fft = (self.hparams.num_freq - 1) * 2
        hop_length = int(self.hparams.frame_shift_ms / 1000 * self.hparams.sample_rate)
        win_length = int(self.hparams.frame_length_ms / 1000 * self.hparams.sample_rate)
        return n_fft, hop_length, win_length

    def _linear_to_mel(self, spectrogram):
        return np.dot(self._mel_basis, spectrogram)

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))

    def _db_to_amp_tf(self, x):
        return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

    def _normalize(self, S):
        return np.clip((S - self.hparams.min_level_db) / -self.hparams.min_level_db, 0, 1)

    def _denormalize_tf(self, S):
        return (tf.clip_by_value(S, 0, 1) * -self.hparams.min_level_db) + self.hparams.min_level_db
