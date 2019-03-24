from error_handling.console_logger import ConsoleLogger

import os
import torch


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
        train_res_recon_errors = []
        train_res_perplexities = []

        sorted_checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split('.')[0].split('_')[-2]))
        for checkpoint_file in sorted_checkpoint_files:
            # Load the checkpoint file
            checkpoint_path = experiment_path + os.sep + checkpoint_file
            ConsoleLogger.status("Loading the checkpoint file '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device_configuration.device)
            train_res_recon_errors += checkpoint['train_res_recon_error']
            train_res_perplexities += checkpoint['train_res_perplexity']

        return train_res_recon_errors, train_res_perplexities
