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
from error_handling.logger_factory import LoggerFactory
from . import LOG_PATH

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import pickle


class VCTKSpeechStream(object):

    def __init__(self, configuration, gpu_ids, use_cuda):
        vctk = VCTK(configuration['data_root'], ratio=configuration['train_val_split'])
        self._training_data = VCTKDataset(vctk.audios_train, vctk.speaker_dic, vctk.utterences, configuration)
        self._validation_data = VCTKDataset(vctk.audios_val, vctk.speaker_dic, vctk.utterences, configuration)
        factor = 1 if len(gpu_ids) == 0 else len(gpu_ids)
        self._training_loader = DataLoader(
            self._training_data,
            batch_size=configuration['batch_size'],
            #batch_size=1,
            shuffle=True,
            num_workers=configuration['num_workers'],
            pin_memory=use_cuda
        )
        self._validation_loader = DataLoader(
            self._validation_data,
            batch_size=configuration['batch_size'],
            #batch_size=1,
            num_workers=configuration['num_workers'],
            pin_memory=use_cuda
        )
        self._speaker_dic = vctk.speaker_dic
        self._train_data_variance = np.var(self._training_data.quantize / 255.0)
        self._logger = LoggerFactory.create(LOG_PATH, __name__)

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

    def export_to_features(self, vctk_path, configuration):
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

        def process(loader, output_dir, input_features_name, output_features_name,
            rate, input_filters_number, output_filters_number, input_target_shape,
            augment_output_features):

            initial_index = 0
            attempts = 10
            current_attempt = 0
            total_length = len(loader)

            while current_attempt < attempts:
                try:
                    i = initial_index
                    bar = tqdm(loader, initial=initial_index)
                    for data in bar:
                        (preprocessed_audio, one_hot, speaker_id, quantized, wav_filename, sampling_rate, shifting_time, random_starting_index, preprocessed_length, top_db) = data

                        output_path = output_dir + os.sep + str(i) + '.pickle'
                        if os.path.isfile(output_path):
                            i += 1
                            bar.set_description('{} already exists'.format(output_path))
                            continue

                        input_features = SpeechFeatures.features_from_name(
                            name=input_features_name,
                            signal=preprocessed_audio,
                            rate=rate,
                            filters_number=input_filters_number
                        )

                        if input_features.shape[0] != input_target_shape[0] or input_features.shape[1] != input_target_shape[1]:
                            ConsoleLogger.warn("Raw features number {} with invalid dimension {} will not be saved. Target shape: {}".format(i, input_features.shape, input_target_shape))
                            i += 1
                            continue

                        output_features = SpeechFeatures.features_from_name(
                            name=output_features_name,
                            signal=preprocessed_audio,
                            rate=rate,
                            filters_number=output_filters_number,
                            augmented=augment_output_features
                        )

                        # TODO: add an option in configuration to save quantized/one_hot or not
                        output = {
                            'preprocessed_audio': preprocessed_audio,
                            'wav_filename': wav_filename,
                            'input_features': input_features,
                            'one_hot': np.array([]),
                            'quantized': np.array([]),
                            'speaker_id': speaker_id,
                            'output_features': output_features,
                            'shifting_time': shifting_time,
                            'random_starting_index': random_starting_index,
                            'preprocessed_length': preprocessed_length,
                            'sampling_rate': sampling_rate,
                            'top_db': top_db
                        }

                        with open(output_path, 'wb') as file:
                            pickle.dump(output, file)

                        bar.set_description('{} saved'.format(output_path))

                        i += 1

                        if i == total_length:
                            bar.update(total_length)
                            break

                    bar.close()
                    break
                except KeyboardInterrupt:
                    bar.close()
                    ConsoleLogger.warn('Keyboard interrupt detected. Leaving the function...')
                    return
                except:
                    error_message = 'An error occured in the data loader at {}/{}. Current attempt: {}/{}'.format(output_dir, i, current_attempt+1, attempts)
                    self._logger.exception(error_message)
                    ConsoleLogger.error(error_message)
                    initial_index = i
                    current_attempt += 1
                    continue

        try:
            ConsoleLogger.status('Processing training part')
            process(
                loader=self._training_loader,
                output_dir=train_features_path,
                input_features_name=configuration['input_features_type'],
                output_features_name=configuration['output_features_type'],
                rate=configuration['sampling_rate'],
                input_filters_number=configuration['input_features_filters'],
                output_filters_number=configuration['output_features_filters'],
                input_target_shape=(configuration['input_features_dim'], configuration['input_features_filters'] * 3),
                augment_output_features=configuration['augment_output_features']
            )
            ConsoleLogger.success('Training part processed')
        except:
            ConsoleLogger.error('An error occured during training features generation')

        try:
            ConsoleLogger.status('Processing validation part')
            process(
                loader=self._validation_loader,
                output_dir=val_features_path,
                input_features_name=configuration['input_features_type'],
                output_features_name=configuration['output_features_type'],
                rate=configuration['sampling_rate'],
                input_filters_number=configuration['input_features_filters'],
                output_filters_number=configuration['output_features_filters'],
                input_target_shape=(configuration['input_features_dim'], configuration['input_features_filters'] * 3),
                augment_output_features=configuration['augment_output_features']
            )
            ConsoleLogger.success('Validation part processed')
        except:
            ConsoleLogger.error('An error occured during validation features generation')
