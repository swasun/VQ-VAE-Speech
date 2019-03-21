from vq_vae_features.features_auto_encoder import FeaturesAutoEncoder
from vq_vae_features.trainer import Trainer as FeaturesTrainer
from error_handling.console_logger import ConsoleLogger
from vq_vae_wavenet.wavenet_auto_encoder import WaveNetAutoEncoder
from vq_vae_wavenet.trainer import Trainer as WaveNetTrainer
from experiments.device_configuration import DeviceConfiguration
from dataset.speech_dataset import SpeechDataset

from torch import nn
import torch.optim as optim
import torch
import os
import yaml


class ModelFactory(object):

    @staticmethod
    def build(configuration, device_configuration, dataset, with_trainer=True):
        ConsoleLogger.status('Building model...')
        if configuration['decoder_type'] == 'deconvolutional':
            auto_encoder = FeaturesAutoEncoder(configuration, device_configuration.device).to(device_configuration.device)
            optimizer = optim.Adam(auto_encoder.parameters(), lr=configuration['learning_rate'], amsgrad=True) # Create an Adam optimizer instance
            if with_trainer:
                trainer = FeaturesTrainer(
                    device_configuration.device,
                    auto_encoder,
                    optimizer,
                    dataset,
                    configuration
                )
        elif configuration['decoder_type'] == 'wavenet':
            auto_encoder = WaveNetAutoEncoder(configuration, dataset.speaker_dic, device_configuration.device).to(device_configuration.device)
            optimizer = optim.Adam(auto_encoder.parameters(), lr=configuration['learning_rate'], amsgrad=True) # Create an Adam optimizer instance
            if with_trainer:
                trainer = WaveNetTrainer(device_configuration.device, auto_encoder, optimizer, dataset, configuration)
        else:
            raise ValueError('Invalid configuration file: there is no decoder_type field')

        auto_encoder = nn.DataParallel(auto_encoder, device_ids=device_configuration.device_ids) if device_configuration.use_data_parallel else auto_encoder

        if with_trainer:
            return auto_encoder, trainer

        return auto_encoder

    @staticmethod
    def load(experiment_path, experiment_name):
        # Check if the specified experiment path exists
        ConsoleLogger.status("Checking if the experiment path '{}' exists".format(experiment_path))
        if not os.path.isdir(experiment_path):
            raise ValueError("Specified experiment path '{}' doesn't not exist".format(experiment_path))

        # List all the files from this directory and raise an error if it's empty
        ConsoleLogger.status('Listing the specified experiment path directory')
        files = os.listdir(experiment_path)
        if not files or len(files) == 0:
            raise ValueError("Specified experiment path '{}' is empty".format(experiment_path))

        # Search the configuration file and the checkpoint files of the specified experiment
        ConsoleLogger.status('Searching the configuration file and the checkpoint files')
        checkpoint_files = list()
        configuration_file = None
        for file in files:
            split_file = file.split('_')
            if len(split_file) > 1 and split_file[0] == experiment_name and split_file[1] == 'configuration.yaml':
                configuration_file = file
            elif len(split_file) > 1 and split_file[0] == experiment_name and split_file[1] != 'configuration.yaml':
                checkpoint_files.append(file)

        # Check if a configuration file was found
        if not configuration_file:
            raise ValueError('No configuration file found with name: {}'.format(experiment_name))

        # Check if at least one checkpoint file was found
        if len(checkpoint_files) == 0:
            raise ValueError('No checkpoint files found with name: {}'.format(experiment_name))

        # Search the latest checkpoint file
        ConsoleLogger.status('Searching the latest checkpoint file')
        latest_checkpoint_file = checkpoint_files[0]
        latest_epoch = int(checkpoint_files[0].split('_')[1])
        for i in range(1, len(checkpoint_files)):
            epoch = int(checkpoint_files[i].split('_')[1])
            if epoch > latest_epoch:
                latest_checkpoint_file = checkpoint_files[i]
                latest_epoch = epoch

        # Load the configuration file
        ConsoleLogger.status('Loading the configuration file')
        configuration = None
        with open(experiment_path + os.sep + configuration_file, 'r') as configuration_file:
            configuration = yaml.load(configuration_file)

        # Update the epoch number to begin with for the future training
        configuration['start_epoch'] = latest_epoch
        
        # Load the device configuration from the configuration state
        device_configuration = DeviceConfiguration.load_from_configuration(configuration)

        # Load the checkpoint file
        checkpoint_path = experiment_path + os.sep + latest_checkpoint_file
        ConsoleLogger.status("Loading the checkpoint file '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device_configuration.device)

        # Load the speech dataset
        ConsoleLogger.status('Loading the speech dataset')
        dataset = SpeechDataset(configuration, device_configuration.gpu_ids, device_configuration.use_cuda)

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
                dataset,
                configuration
            )
        # Else if the decoder is a wavenet
        elif configuration['decoder_type'] == 'wavenet':
            # Create the model and map it to the specified device
            model = WaveNetAutoEncoder(configuration, dataset.speaker_dic, device_configuration.device).to(device_configuration.device)

            # Load the model and optimizer state dicts
            model, optimizer = load_state_dicts(model, checkpoint)

            # Create a trainer instance associated with our model
            trainer = WaveNetTrainer(
                device_configuration.device,
                model,
                optimizer,
                dataset,
                configuration
            )
        else:
            raise ValueError('Invalid configuration file: there is no decoder_type field')

        # Use data parallelization if needed and available
        model = nn.DataParallel(model, device_ids=device_configuration.device_ids) if device_configuration.use_data_parallel else model

        return model, trainer, configuration, dataset
