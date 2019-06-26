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
from error_handling.console_logger import ConsoleLogger
from error_handling.logger_factory import LoggerFactory
from . import LOG_PATH

from torch.utils.data import DataLoader
import numpy as np
import pathlib
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


class VCTKFeaturesStream(object):

    def __init__(self, vctk_path, configuration, gpu_ids, use_cuda):
        self._normalizer = None
        if configuration['normalize']:
            with open(configuration['normalizer_path'], 'rb') as file:
                self._normalizer = pickle.load(file)

        self._training_data = VCTKFeaturesDataset(vctk_path, 'train', self._normalizer, features_path=configuration['features_path'])
        self._validation_data = VCTKFeaturesDataset(vctk_path, 'val', self._normalizer, features_path=configuration['features_path'])
        factor = 1 if len(gpu_ids) == 0 else len(gpu_ids)

        factor = 1
        #configuration['batch_size'] = 1
        self._training_batch_size = configuration['batch_size']
        self._validation_batch_size = 1

        self._training_loader = DataLoader(
            self._training_data,
            batch_size=configuration['batch_size'],
            shuffle=True,
            num_workers=configuration['num_workers'],
            pin_memory=use_cuda
        )
        self._validation_loader = DataLoader(
            self._validation_data,
            #batch_size=configuration['batch_size'],
            batch_size=self._validation_batch_size,
            num_workers=configuration['num_workers'],
            pin_memory=use_cuda
        )
        self._speaker_dic = self._make_speaker_dic(vctk_path + os.sep + 'raw' + os.sep + 'VCTK-Corpus')
        self._vctk_path = vctk_path
        self._logger = LoggerFactory.create(LOG_PATH, __name__)
        self._normalizer_path = configuration['normalizer_path']

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
    def training_batch_size(self):
        return self._training_batch_size

    @property
    def validation_batch_size(self):
        return self._validation_batch_size

    @property
    def normalizer(self):
        return self._normalizer

    def _make_speaker_dic(self, root):
        speakers = [
            str(speaker.name) for speaker in pathlib.Path(root).glob('wav48/*/')]
        speakers = sorted([speaker for speaker in speakers])
        speaker_dic = {speaker: i for i, speaker in enumerate(speakers)}
        return speaker_dic

    def compute_dataset_stats(self):
        initial_index = 0
        attempts = 10
        current_attempt = 0
        total_length = len(self._training_loader)
        train_mfccs = list()
        while current_attempt < attempts:
            try:
                i = initial_index
                train_bar = tqdm(self._training_loader, initial=initial_index)
                for data in train_bar:
                    input_features = data['input_features']
                    train_mfccs.append(input_features.detach().view(input_features.size(1), input_features.size(2)).numpy())

                    i += 1

                    if i == total_length:
                        train_bar.update(total_length)
                        break

                train_bar.close()
                break

            except KeyboardInterrupt:
                train_bar.close()
                ConsoleLogger.warn('Keyboard interrupt detected. Leaving the function...')
                return
            except:
                error_message = 'An error occured in the data loader at {}. Current attempt: {}/{}'.format(i, current_attempt+1, attempts)
                self._logger.exception(error_message)
                ConsoleLogger.error(error_message)
                initial_index = i
                current_attempt += 1
                continue


        ConsoleLogger.status('Compute mean of mfccs training set...')
        train_mean = np.concatenate(train_mfccs).mean(axis=0)

        ConsoleLogger.status('Compute std of mfccs training set...')
        train_std = np.concatenate(train_mfccs).std(axis=0)

        stats = {
            'train_mean': train_mean,
            'train_std': train_std
        }

        ConsoleLogger.status('Writing stats in file...')
        with open(self._normalizer_path, 'wb') as file: # TODO: do not use hardcoded path
            pickle.dump(stats, file)

        train_mfccs_norm = (train_mfccs[0] - train_mean) / train_std

        ConsoleLogger.status('Computing example plot...')
        _, axs = plt.subplots(2, sharex=True)
        axs[0].imshow(train_mfccs[0].T, aspect='auto', origin='lower')
        axs[0].set_ylabel('Unnormalized')
        axs[1].imshow(train_mfccs_norm.T, aspect='auto', origin='lower')
        axs[1].set_ylabel('Normalized')
        plt.savefig('mfcc_normalization_comparaison.png') # TODO: do not use hardcoded path
