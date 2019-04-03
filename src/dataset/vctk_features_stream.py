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

from dataset.vctk_features_dataset import VCTKFeaturesDataset

from torch.utils.data import DataLoader
import numpy as np
import pathlib
import os


class VCTKFeaturesStream(object):

    def __init__(self, vctk_path, configuration, gpu_ids, use_cuda):
        self._training_data = VCTKFeaturesDataset(vctk_path, 'train')
        self._validation_data = VCTKFeaturesDataset(vctk_path, 'val')
        factor = 1 if len(gpu_ids) == 0 else len(gpu_ids)
        self._training_loader = DataLoader(
            self._training_data,
            batch_size=configuration['batch_size'] * factor,
            shuffle=True,
            num_workers=configuration['num_workers'],
            pin_memory=use_cuda
        )
        self._validation_loader = DataLoader(
            self._validation_data,
            batch_size=configuration['batch_size'] * factor,
            num_workers=configuration['num_workers'],
            pin_memory=use_cuda
        )
        self._speaker_dic = self._make_speaker_dic(vctk_path + os.sep + 'VCTK-Corpus')

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

    def _make_speaker_dic(self, root):
        speakers = [
            str(speaker.name) for speaker in pathlib.Path(root).glob('wav48/*/')]
        speakers = sorted([speaker for speaker in speakers])
        speaker_dic = {speaker: i for i, speaker in enumerate(speakers)}
        return speaker_dic
