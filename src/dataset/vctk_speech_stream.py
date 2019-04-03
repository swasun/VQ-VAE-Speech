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

from dataset.vctk_dataset import VCTKDataset
from dataset.vctk import VCTK
from vq_vae_speech.speech_features import SpeechFeatures
from error_handling.console_logger import ConsoleLogger

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import pickle


class VCTKSpeechStream(object):

    def __init__(self, configuration, gpu_ids, use_cuda):
        vctk = VCTK(configuration['data_root'], ratio=configuration['train_val_split'])
        self._training_data = VCTKDataset(vctk.audios_train, vctk.speaker_dic, configuration)
        self._validation_data = VCTKDataset(vctk.audios_val, vctk.speaker_dic, configuration)
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

    def export_to_features(self, vctk_path, raw_feature_name='mfcc', quantized_features_name='logfbank', rate=16000, filters_number=13):
        if not os.path.isdir(vctk_path):
            raise ValueError("VCTK dataset not found at path '{}'".format(vctk_path))

        # Create the features path directory if it doesn't exist
        features_path = vctk_path + os.sep + 'features'
        if not os.path.isdir(features_path):
            ConsoleLogger.status('Creating features directory at path: {}'.format(features_path))
            os.mkdir(features_path)
        else:
            ConsoleLogger.status('Features directory already created at path: {}'.format(features_path))

        # Create the features path directory if it doesn't exist
        train_features_path = features_path + os.sep + 'train'
        if not os.path.isdir(train_features_path):
            ConsoleLogger.status('Creating train features directory at path: {}'.format(train_features_path))
            os.mkdir(train_features_path)
        else:
            ConsoleLogger.status('Train features directory already created at path: {}'.format(train_features_path))

        # Create the features path directory if it doesn't exist
        val_features_path = features_path + os.sep + 'val'
        if not os.path.isdir(val_features_path):
            ConsoleLogger.status('Creating val features directory at path: {}'.format(val_features_path))
            os.mkdir(val_features_path)
        else:
            ConsoleLogger.status('Val features directory already created at path: {}'.format(val_features_path))

        def process(loader, output_path, raw_feature_name, quantized_features_name, rate, filters_number, target_shape):
            bar = tqdm(loader)
            i = 0
            for data in bar:
                (raw, one_hot, speaker_id, quantized) = data

                raw_features = SpeechFeatures.features_from_name(
                    name=raw_feature_name,
                    signal=raw,
                    rate=rate,
                    filters_number=filters_number
                )

                if raw_features.shape[0] != target_shape[0] or raw_features.shape[1] != target_shape[1]:
                    ConsoleLogger.warn("Raw features number {} with invalid dimension {} will not be saved. Target shape: {}".format(i, raw_features.shape, target_shape))
                    i += 1
                    continue

                quantized_features = SpeechFeatures.features_from_name(
                    name=quantized_features_name,
                    signal=quantized,
                    rate=rate,
                    filters_number=filters_number
                )

                output = {
                    'raw_features': raw_features,
                    'one_hot': one_hot,
                    'speaker_id': speaker_id,
                    'quantized_features': quantized_features
                }

                with open(output_path + os.sep + str(i) + '.pickle', 'wb') as file:
                    pickle.dump(output, file)

                i += 1

        ConsoleLogger.status('Processing training part')
        process(
            self._training_loader,
            train_features_path,
            raw_feature_name,
            quantized_features_name,
            rate,
            filters_number,
            (47, 39) # TODO: move it in the configuration
        )
        ConsoleLogger.success('Training part processed')

        ConsoleLogger.status('Processing validation part')
        process(
            self._validation_loader,
            val_features_path,
            raw_feature_name,
            quantized_features_name,
            rate,
            filters_number,
            (47, 39) # TODO: move it in the configuration
        )
        ConsoleLogger.success('Validation part processed')
