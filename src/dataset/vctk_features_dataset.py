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

from torch.utils.data import Dataset
import pickle
import os
import numpy as np


class VCTKFeaturesDataset(Dataset):

    def __init__(self, vctk_path, subdirectory, normalizer=None, features_path='features'):
        self._vctk_path = vctk_path
        self._subdirectory = subdirectory
        features_path = self._vctk_path + os.sep + features_path
        self._sub_features_path = features_path + os.sep + self._subdirectory
        self._files_number = len(os.listdir(self._sub_features_path))
        self._normalizer = normalizer

    def __getitem__(self, index):
        dic = None
        path = self._sub_features_path + os.sep + str(index) + '.pickle'

        if not os.path.isfile(path):
            raise OSError("No such file '{}'".format(path))

        if os.path.getsize(path) == 0:
            raise OSError("Empty file '{}'".format(path))

        with open(path, 'rb') as file:
            dic = pickle.load(file)

        if self._normalizer:
            dic['input_features'] = (dic['input_features'] - self._normalizer['train_mean']) / self._normalizer['train_std']
            dic['output_features'] = (dic['output_features'] - self._normalizer['train_mean']) / self._normalizer['train_std']

        dic['quantized'] = np.array([]) if dic['quantized'] is None else dic['quantized']
        dic['one_hot'] = np.array([]) if dic['one_hot'] is None else dic['one_hot']
        dic['index'] = index

        return dic

    def __len__(self):
        return self._files_number
