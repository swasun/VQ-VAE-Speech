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

from error_handling.console_logger import ConsoleLogger
from experiments.device_configuration import DeviceConfiguration

import os
import torch
import yaml


class CheckpointUtils(object):

    @staticmethod
    def search_configuration_and_checkpoints_files(experiment_path, experiment_name):
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
            # Check if the file is a checkpoint or config file by looking at the extension
            if not '.pth' in file and '.yaml' not in file:
                continue
            split_file = file.split('_')
            if len(split_file) > 1 and split_file[0] == experiment_name and split_file[1] == 'configuration.yaml':
                configuration_file = file
            elif len(split_file) > 1 and split_file[0] == experiment_name and split_file[1] != 'configuration.yaml':
                checkpoint_files.append(file)

        return configuration_file, checkpoint_files

    @staticmethod
    def search_latest_checkpoint_file(checkpoint_files):
        # Search the latest checkpoint file
        ConsoleLogger.status('Searching the latest checkpoint file')
        latest_checkpoint_file = checkpoint_files[0]
        latest_epoch = int(checkpoint_files[0].split('_')[1])
        for i in range(1, len(checkpoint_files)):
            epoch = int(checkpoint_files[i].split('_')[1])
            if epoch > latest_epoch:
                latest_checkpoint_file = checkpoint_files[i]
                latest_epoch = epoch

        return latest_checkpoint_file, latest_epoch

    @staticmethod
    def merge_experiment_losses(experiment_path, checkpoint_files, device_configuration):
        train_res_losses = {}
        train_res_perplexities = []

        sorted_checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split('.')[0].split('_')[-2]))
        for checkpoint_file in sorted_checkpoint_files:
            # Load the checkpoint file
            checkpoint_path = experiment_path + os.sep + checkpoint_file
            ConsoleLogger.status("Loading the checkpoint file '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device_configuration.device)
            for loss_entry in checkpoint['train_res_recon_error']:
                for key in loss_entry.keys():
                    if key not in train_res_losses:
                        train_res_losses[key] = list()
                    train_res_losses[key].append(loss_entry[key])
            train_res_perplexities += checkpoint['train_res_perplexity']

        return train_res_losses, train_res_perplexities

    @staticmethod
    def retreive_losses_values(experiment_path, experiment):
        experiment_name = experiment.name

        ConsoleLogger.status("Searching configuration and checkpoints of experiment '{}' at path '{}'".format(experiment_name, experiment_path))
        configuration_file, checkpoint_files = CheckpointUtils.search_configuration_and_checkpoints_files(
            experiment_path,
            experiment_name
        )

        # Check if a configuration file was found
        if not configuration_file:
            raise ValueError('No configuration file found with name: {}'.format(experiment_name))

        # Check if at least one checkpoint file was found
        if len(checkpoint_files) == 0:
            raise ValueError('No checkpoint files found with name: {}'.format(experiment_name))

        # Load the configuration file
        configuration_path = experiment_path + os.sep + configuration_file
        ConsoleLogger.status("Loading the configuration file '{}'".format(configuration_path))
        configuration = None
        with open(configuration_path, 'r') as file:
            configuration = yaml.load(file, Loader=yaml.FullLoader)
        
        # Load the device configuration from the configuration state
        device_configuration = DeviceConfiguration.load_from_configuration(configuration)

        ConsoleLogger.status("Merge {} checkpoint losses of experiment '{}'".format(len(checkpoint_files), experiment_name))
        train_res_losses, train_res_perplexities = CheckpointUtils.merge_experiment_losses(
            experiment_path,
            checkpoint_files,
            device_configuration
        )

        return train_res_losses, train_res_perplexities, len(checkpoint_files)
