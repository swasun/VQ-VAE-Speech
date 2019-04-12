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

from experiments.device_configuration import DeviceConfiguration
from experiments.checkpoint_utils import CheckpointUtils
from vq_vae_features.features_auto_encoder import FeaturesAutoEncoder
from vq_vae_features.trainer import Trainer as FeaturesTrainer
from vq_vae_features.evaluator import Evaluator as FeaturesEvaluator
from vq_vae_wavenet.wavenet_auto_encoder import WaveNetAutoEncoder
from vq_vae_wavenet.trainer import Trainer as WaveNetTrainer
from vq_vae_wavenet.evaluator import Evaluator as WaveNetEvaluator
from error_handling.console_logger import ConsoleLogger
from dataset.vctk_features_stream import VCTKFeaturesStream

from torch import nn
import torch.optim as optim
import torch
import os
import yaml


class ModelFactory(object):

    @staticmethod
    def build(configuration, device_configuration, data_stream, with_trainer=True):
        ConsoleLogger.status('Building model...')
        if configuration['decoder_type'] == 'deconvolutional':
            model = FeaturesAutoEncoder(configuration, device_configuration.device).to(device_configuration.device)
            optimizer = optim.Adam(model.parameters(), lr=configuration['learning_rate'], amsgrad=True) # Create an Adam optimizer instance
            if with_trainer:
                trainer = FeaturesTrainer(
                    device_configuration.device,
                    model,
                    optimizer,
                    data_stream,
                    configuration
                )
                evaluator = FeaturesEvaluator(
                    device_configuration.device,
                    model,
                    data_stream,
                    configuration
                )
        elif configuration['decoder_type'] == 'wavenet':
            model = WaveNetAutoEncoder(configuration, data_stream.speaker_dic, device_configuration.device).to(device_configuration.device)
            optimizer = optim.Adam(model.parameters(), lr=configuration['learning_rate'], amsgrad=True) # Create an Adam optimizer instance
            if with_trainer:
                trainer = WaveNetTrainer(device_configuration.device, model, optimizer, data_stream, configuration)
                evaluator = WaveNetEvaluator(device_configuration.device, model, data_stream)
        else:
            raise ValueError('Invalid configuration file: there is no decoder_type field')

        model = nn.DataParallel(model, device_ids=device_configuration.gpu_ids) if device_configuration.use_data_parallel else model

        if with_trainer:
            return model, trainer, evaluator

        return model

    @staticmethod
    def load(experiment_path, experiment_name, data_path='../data'):
        configuration_file, checkpoint_files = CheckpointUtils.search_configuration_and_checkpoints_files(experiment_path, experiment_name)

        # Check if a configuration file was found
        if not configuration_file:
            raise ValueError('No configuration file found with name: {}'.format(experiment_name))

        # Check if at least one checkpoint file was found
        if len(checkpoint_files) == 0:
            ConsoleLogger.warn('No checkpoint files found with name: {}'.format(experiment_name))
            return [configuration_file]

        latest_checkpoint_file, latest_epoch = CheckpointUtils.search_latest_checkpoint_file(checkpoint_files)

        # Load the configuration file
        ConsoleLogger.status('Loading the configuration file')
        configuration = None
        with open(experiment_path + os.sep + configuration_file, 'r') as file:
            configuration = yaml.load(file)

        # Update the epoch number to begin with for the future training
        configuration['start_epoch'] = latest_epoch

        # Load the device configuration from the configuration state
        device_configuration = DeviceConfiguration.load_from_configuration(configuration)

        # Load the checkpoint file
        checkpoint_path = experiment_path + os.sep + latest_checkpoint_file
        ConsoleLogger.status("Loading the checkpoint file '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device_configuration.device)

        # Load the data stream
        ConsoleLogger.status('Loading the data stream')
        data_stream = VCTKFeaturesStream(data_path + os.sep + 'vctk', configuration, device_configuration.gpu_ids, device_configuration.use_cuda)

        def load_state_dicts(model, checkpoint):
            # Load the state dict from the checkpoint to the model
            model.load_state_dict(checkpoint['model'])
            # Create an Adam optimizer using the model parameters
            optimizer = optim.Adam(model.parameters())
            # Load the state dict from the checkpoint to the optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Map the optimizer memory into the specified device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device_configuration.device)
            return model, optimizer

        # If the decoder type is a deconvolutional
        if configuration['decoder_type'] == 'deconvolutional':
            # Create the model and map it to the specified device
            model = FeaturesAutoEncoder(configuration, device_configuration.device).to(device_configuration.device)

            # Load the model and optimizer state dicts
            model, optimizer = load_state_dicts(model, checkpoint)

            # Create a trainer instance associated with our model
            trainer = FeaturesTrainer(
                device_configuration.device,
                model,
                optimizer,
                data_stream,
                configuration
            )

            # Create a evaluator instance associated with our model
            evaluator = FeaturesEvaluator(
                device_configuration.device,
                model,
                data_stream,
                configuration
            )
        # Else if the decoder is a wavenet
        elif configuration['decoder_type'] == 'wavenet':
            # Create the model and map it to the specified device
            model = WaveNetAutoEncoder(configuration, data_stream.speaker_dic, device_configuration.device).to(device_configuration.device)

            # Load the model and optimizer state dicts
            model, optimizer = load_state_dicts(model, checkpoint)

            # Create a trainer instance associated with our model
            trainer = WaveNetTrainer(
                device_configuration.device,
                model,
                optimizer,
                data_stream,
                configuration
            )

            # Create a evaluator instance associated with our model
            evaluator = WaveNetEvaluator(
                device_configuration.device,
                model,
                data_stream
            )
        else:
            raise ValueError('Invalid configuration file: there is no decoder_type field')

        # Use data parallelization if needed and available
        model = nn.DataParallel(model, device_ids=device_configuration.gpu_ids) if device_configuration.use_data_parallel else model

        return [model, trainer, evaluator, configuration, data_stream]
