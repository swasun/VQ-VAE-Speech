 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-Speech.                                               #
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

from dataset.vctk import VCTK
from speech_utils.mu_law import MuLaw

from torch.utils.data import Dataset
import numpy as np
import random
import pathlib
import librosa
import os
import textgrid


class VCTKDataset(Dataset):

    def __init__(self, audios, speaker_dic, utterences, configuration):
        self._audios = audios
        self._speaker_dic = speaker_dic
        self._utterences = utterences
        self._sampling_rate = configuration['sampling_rate']
        self._res_type = configuration['res_type']
        self._top_db = configuration['top_db']
        self._length = None if configuration['length'] is None else configuration['length'] + 1
        self._quantize = configuration['quantize']

    def _preprocessing(self, audio, quantized):
        if self._length is not None:
            if len(audio) <= self._length :
                # padding
                pad = self._length - len(audio)
                audio = np.concatenate(
                    (audio, np.zeros(pad, dtype=np.float32)))
                quantized = np.concatenate(
                    (quantized, self._quantize // 2 * np.ones(pad)))
                quantized = quantized.astype(np.long)
                start_trimming = None
            else:
                # trimming
                start_trimming = random.randint(0, len(audio) - self._length - 1)
                audio = audio[start_trimming:start_trimming + self._length]
                quantized = quantized[start_trimming:start_trimming + self._length]

        # ont_hot for input
        one_hot = np.identity(
            self._quantize,
            dtype=np.float32
        )[quantized]
        one_hot = np.expand_dims(one_hot.T, 2)

        audio = np.expand_dims(audio, 0) # expand channel
        audio = np.expand_dims(audio, -1) # expand height

        # target
        quantized = np.expand_dims(quantized, 1)

        return audio, one_hot[:, :-1], quantized[1:], start_trimming

    @staticmethod
    def preprocess_audio(audio, length, expand_dims=False):
        if length is not None:
            if len(audio) <= length :
                # padding
                pad = length - len(audio)
                audio = np.concatenate(
                    (audio, np.zeros(pad, dtype=np.float32)))
            else:
                # triming
                start = random.randint(0, len(audio) -length  - 1)
                audio = audio[start:start + length]

        if expand_dims:
            audio = np.expand_dims(audio, 0) # expand channel
            audio = np.expand_dims(audio, -1) # expand height

        return audio

    def __getitem__(self, index):
        wav_filename = self._audios[index]

        # Check if a groundtruth is available
        split_path = wav_filename.split(os.sep)
        groundtruth_alignment_path = os.sep.join(split_path[:-3]) + os.sep + 'phonemes' + os.sep + split_path[-2] + os.sep + split_path[-1].replace('.wav', '.TextGrid')
        detected_sil_duration = 0.0
        if os.path.isfile(groundtruth_alignment_path):
            tg = textgrid.TextGrid()
            tg.read(groundtruth_alignment_path)
            for interval in tg.tiers[1]:
                if interval.mark != 'sil':
                    break
                detected_sil_duration += float(interval.maxTime) - float(interval.minTime)
 
        audio, trimming_time = self._load_wav(
            wav_filename,
            self._sampling_rate,
            self._res_type,
            self._top_db,
            trimming_duration=detected_sil_duration if detected_sil_duration != 0.0 else None
        )

        quantized = MuLaw.encode(audio)

        speaker = pathlib.Path(wav_filename).parent.name

        speaker_id = np.array(self._speaker_dic[speaker], dtype=np.long)

        preprocessed_audio, one_hot, quantized, start_trimming = self._preprocessing(audio, quantized)

        shifting_time = trimming_time + (0 if start_trimming is None else start_trimming / self._sampling_rate)

        return preprocessed_audio, one_hot, speaker_id, quantized, wav_filename, self._sampling_rate, \
            shifting_time, 0 if start_trimming is None else start_trimming, self._length - 1, self._top_db

    def __len__(self):
        return len(self._audios)

    def _load_wav(self, filename, sampling_rate, res_type, top_db, trimming_duration=None):
        raw, _ = librosa.load(filename, sampling_rate, res_type=res_type)
        if trimming_duration is None:
            trimmed_audio, trimming_indices = librosa.effects.trim(raw, top_db=top_db)
            trimming_time = trimming_indices[0] / sampling_rate
        else:
            trimmed_audio = raw[int(trimming_duration * sampling_rate):]
            trimming_time = trimming_duration
        trimmed_audio /= np.abs(trimmed_audio).max()
        trimmed_audio = trimmed_audio.astype(np.float32)

        return trimmed_audio, trimming_time

    @property
    def speaker_dic(self):
        return self._speaker_dic

    @property
    def quantize(self):
        return self._quantize

    @property
    def utterences(self):
        return self._utterences

