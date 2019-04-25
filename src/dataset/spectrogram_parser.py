 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (c) 2017 Sean Naren                                                     #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from dataset.audio_parser import AudioParser
from dataset.audio_loader import AudioLoader
from dataset.noise_injector import NoiseInjector

import torch
import scipy
import numpy as np
import librosa
from tempfile import NamedTemporaryFile
import os


class SpectrogramParser(AudioParser):

    default_audio_conf = {
        'window_size': 0.02,
        'window_stride': 0.01, # timestep
        'noise_prob': 0.4,
        'sample_rate': 16000,
        'noise_dir': None,
        'noise_levels': (0.0, 0.5),
        'window': 'hamming'
    }

    def __init__(self, audio_conf=default_audio_conf, normalize=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment
        self.noiseInjector = NoiseInjector(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')

    def parse_audio_from_file(self, audio_path):
        if self.augment:
            y = self._load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = AudioLoader.load(audio_path)
        return self.parse_audio(y)

    def parse_audio(self, y):
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, _ = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError

    def _load_randomly_augmented_audio(self, path, sample_rate=16000, tempo_range=(0.85, 1.15),
        gain_range=(-6, 8)):
        """
        Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
        Returns the augmented utterance.
        """
        low_tempo, high_tempo = tempo_range
        tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
        low_gain, high_gain = gain_range
        gain_value = np.random.uniform(low=low_gain, high=high_gain)
        audio = self._augment_audio_with_sox(path=path, sample_rate=sample_rate,
            tempo=tempo_value, gain=gain_value)
        return audio

    def _augment_audio_with_sox(self, path, sample_rate, tempo, gain):
        """
        Changes tempo and gain of the recording with sox and loads it.
        """
        with NamedTemporaryFile(suffix=".wav") as augmented_file:
            augmented_filename = augmented_file.name
            sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
            sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                augmented_filename,
                " ".join(sox_augment_params))
            os.system(sox_params)
            y = AudioLoader.load(augmented_filename)
            return y
