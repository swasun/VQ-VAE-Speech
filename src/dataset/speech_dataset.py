from dataset.vctk_dataset import VCTKDataset
from dataset.vctk import VCTK

from torch.utils.data import DataLoader
import numpy as np


class SpeechDataset(object):

    def __init__(self, params, gpu_ids, use_cuda):
        vctk = VCTK(params['data_root'], ratio=params['train_val_split'])
        self._training_data = VCTKDataset(vctk.audios_train, vctk.speaker_dic, params)
        self._validation_data = VCTKDataset(vctk.audios_val, vctk.speaker_dic, params)
        self._training_loader = DataLoader(self._training_data, batch_size=params['batch_size'] * len(gpu_ids), shuffle=True,
                                num_workers=params['num_workers'], pin_memory=use_cuda)
        self._validation_loader = DataLoader(self._validation_data, batch_size=1, num_workers=params['num_workers'], pin_memory=use_cuda)
        self._speaker_dic = vctk.speaker_dic
        self._train_data_variance = np.var(self._training_data.quantize / 255.0)

    @property
    def training_data(self):
        return self._training_data

    @property
    def validation_data(self):
        return self._validation_data

    @property
    def training_loader(self):
        return self._training_loader

    @property
    def validation_loader(self):
        return self._validation_loader

    @property
    def speaker_dic(self):
        return self._speaker_dic

    @property
    def train_data_variance(self):
        return self._train_data_variance
