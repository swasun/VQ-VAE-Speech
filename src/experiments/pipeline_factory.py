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
from experiments.convolutional_trainer import ConvolutionalTrainer
from experiments.evaluator import Evaluator
from models.convolutional_vq_vae import ConvolutionalVQVAE
from error_handling.console_logger import ConsoleLogger
from dataset.vctk_features_stream import VCTKFeaturesStream

from torch import nn
import torch.optim as optim
import torch
import os
import yaml


class PipelineFactory(object):

    @staticmethod
    def build(configuration, device_configuration, experiments_path, experiment_name, results_path):
        data_stream = VCTKFeaturesStream('../data/vctk', configuration, device_configuration.gpu_ids, device_configuration.use_cuda)

        if configuration['decoder_type'] == 'deconvolutional':
            vqvae_model = ConvolutionalVQVAE(configuration, device_configuration.device).to(device_configuration.device)
            evaluator = Evaluator(device_configuration.device, vqvae_model, data_stream, configuration,
                results_path, experiment_name)
        else:
            raise NotImplementedError("Decoder type '{}' isn't implemented for now".format(configuration['decoder_type']))

        if configuration['trainer_type'] == 'convolutional':
            trainer = ConvolutionalTrainer(device_configuration.device, data_stream,
                configuration, experiments_path, experiment_name, **{'model': vqvae_model})
        else:
            raise NotImplementedError("Trainer type '{}' isn't implemented for now".format(configuration['trainer_type']))

        vqvae_model = nn.DataParallel(vqvae_model, device_ids=device_configuration.gpu_ids) if device_configuration.use_data_parallel else vqvae_model

        return trainer, evaluator

    @staticmethod
    def load_configuration_and_checkpoints(experiments_path, experiment_name):
        configuration_file, checkpoint_files = CheckpointUtils.search_configuration_and_checkpoints_files(
            experiments_path, experiment_name)

        # Check if a configuration file was found
        if not configuration_file:
            raise ValueError('No configuration file found with name: {}'.format(experiment_name))

        # Check if at least one checkpoint file was found
        if len(checkpoint_files) == 0:
            ConsoleLogger.warn('No checkpoint files found with name: {}'.format(experiment_name))

        return configuration_file, checkpoint_files

    @staticmethod
    def load(experiments_path, experiment_name, results_path, data_path='../data'):
        error_caught = False

        try:
            configuration_file, checkpoint_files = PipelineFactory.load_configuration_and_checkpoints(
                experiments_path, experiment_name)
        except:
            ConsoleLogger.error('Failed to load existing configuration. Building a new model...')
            error_caught = True

        # Load the configuration file
        ConsoleLogger.status('Loading the configuration file')
        configuration = None
        with open(experiments_path + os.sep + configuration_file, 'r') as file:
            configuration = yaml.load(file, Loader=yaml.FullLoader)
        device_configuration = DeviceConfiguration.load_from_configuration(configuration)

        if error_caught or len(checkpoint_files) == 0:
            trainer, evaluator = PipelineFactory.build(configuration, device_configuration, experiments_path, experiment_name, results_path)
        else:
            latest_checkpoint_file, latest_epoch = CheckpointUtils.search_latest_checkpoint_file(checkpoint_files)
            # Update the epoch number to begin with for the future training
            configuration['start_epoch'] = latest_epoch

            # Load the checkpoint file
            checkpoint_path = experiments_path + os.sep + latest_checkpoint_file
            ConsoleLogger.status("Loading the checkpoint file '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device_configuration.device)

            # Load the data stream
            ConsoleLogger.status('Loading the data stream')
            data_stream = VCTKFeaturesStream(data_path + os.sep + 'vctk', configuration, device_configuration.gpu_ids, device_configuration.use_cuda)

            def load_state_dicts(model, checkpoint, model_name, optimizer_name):
                # Load the state dict from the checkpoint to the model
                model.load_state_dict(checkpoint[model_name])
                # Create an Adam optimizer using the model parameters
                optimizer = optim.Adam(model.parameters())
                # Load the state dict from the checkpoint to the optimizer
                optimizer.load_state_dict(checkpoint[optimizer_name])
                # Map the optimizer memory into the specified device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device_configuration.device)
                return model, optimizer

            # If the decoder type is a deconvolutional
            if configuration['decoder_type'] == 'deconvolutional':
                # Create the model and map it to the specified device
                vqvae_model = ConvolutionalVQVAE(configuration, device_configuration.device).to(device_configuration.device)
                evaluator = Evaluator(device_configuration.device, vqvae_model, data_stream,
                    configuration, results_path, experiment_name)

                # Load the model and optimizer state dicts
                vqvae_model, vqvae_optimizer = load_state_dicts(vqvae_model, checkpoint, 'model', 'optimizer')
            else:
                raise NotImplementedError("Decoder type '{}' isn't implemented for now".format(configuration['decoder_type']))

            # Temporary backward compatibility
            if 'trainer_type' not in configuration:
                ConsoleLogger.error("trainer_type was not found in configuration file. Use 'convolutional' by default.")
                configuration['trainer_type'] = 'convolutional' 

            if configuration['trainer_type'] == 'convolutional':
                trainer = ConvolutionalTrainer(device_configuration.device, data_stream,
                    configuration, experiments_path, experiment_name, **{'model': vqvae_model, 
                    'optimizer': vqvae_optimizer})
            else:
                raise NotImplementedError("Trainer type '{}' isn't implemented for now".format(configuration['trainer_type']))

            # Use data parallelization if needed and available
            vqvae_model = nn.DataParallel(vqvae_model, device_ids=device_configuration.gpu_ids) \
                if device_configuration.use_data_parallel else vqvae_model

        return trainer, evaluator, configuration, device_configuration
