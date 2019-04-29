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
from vq_vae_speech.mu_law import MuLaw

from torch.utils.data import Dataset
import numpy as np
import random
import pathlib
import librosa


class VCTKDataset(Dataset):

    def __init__(self, audios, speaker_dic, configuration):
        self._audios = audios
        self._speaker_dic = speaker_dic
        self._sampling_rate = configuration['sampling_rate']
        self._res_type = configuration['res_type']
        self._top_db = configuration['top_db']
        self._length = None if configuration['length'] is None else configuration['length'] + 1
        self._quantize = configuration['quantize']

    def _preprocessing(self, raw, quantized):
        if self._length is not None:
            if len(raw) <=self._length :
                # padding
                pad = self._length - len(raw)
                raw = np.concatenate(
                    (raw, np.zeros(pad, dtype=np.float32)))
                quantized = np.concatenate(
                    (quantized, self._quantize // 2 * np.ones(pad)))
                quantized = quantized.astype(np.long)
            else:
                # triming
                start = random.randint(0, len(raw) -self._length  - 1)
                raw = raw[start:start + self._length ]
                quantized = quantized[start:start + self._length ]

        # ont_hot for input
        one_hot = np.identity(
            self._quantize,
            dtype=np.float32
        )[quantized]
        one_hot = np.expand_dims(one_hot.T, 2)

        raw = np.expand_dims(raw, 0) # expand channel
        raw = np.expand_dims(raw, -1) # expand height

        # target
        quantized = np.expand_dims(quantized, 1)

        return raw, one_hot[:, :-1], quantized[1:]

    @staticmethod
    def preprocessing_raw(raw, length, expand_dims=False):
        if length is not None:
            if len(raw) <= length :
                # padding
                pad = length - len(raw)
                raw = np.concatenate(
                    (raw, np.zeros(pad, dtype=np.float32)))
            else:
                # triming
                start = random.randint(0, len(raw) -length  - 1)
                raw = raw[start:start + length ]

        if expand_dims:
            raw = np.expand_dims(raw, 0) # expand channel
            raw = np.expand_dims(raw, -1) # expand height

        return raw

    def __getitem__(self, index):
        wav_filename = self._audios[index]
        raw = self._load_wav(wav_filename, self._sampling_rate, self._res_type, self._top_db)

        quantized = MuLaw.encode(raw)

        speaker = pathlib.Path(wav_filename).parent.name

        speaker_id = np.array(self._speaker_dic[speaker], dtype=np.long)

        raw, one_hot, quantized = self._preprocessing(raw, quantized)

        return raw, one_hot, speaker_id, quantized, wav_filename

    def __len__(self):
        return len(self._audios)

    def _load_wav(self, filename, sampling_rate, res_type, top_db):
        raw, _ = librosa.load(filename, sampling_rate, res_type=res_type)
        raw, _ = librosa.effects.trim(raw, top_db=top_db)
        raw /= np.abs(raw).max()
        raw = raw.astype(np.float32)

        return raw

    @property
    def speaker_dic(self):
        return self._speaker_dic

    @property
    def quantize(self):
        return self._quantize


if __name__ =='__main__':

    from torch.utils.data import DataLoader

    vctk = VCTK('./')
    configuration = { 'length':7680, 'quantize':256, 'sampling_rate':16000, 'res_type':'kaiser_fast', 'top_db':20 }
    train_dataset = VCTKDataset(vctk.audios_train, vctk.speaker_dic, configuration)
    val_dataset = VCTKDataset(vctk.audios_val, vctk.speaker_dic, configuration)

    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)
    raw, one_hot, speaker_id, quantized = next(iter(train_loader))
    raw_val, one_hot_val, speaker_id_val, quantized_val = next(iter(val_loader))

