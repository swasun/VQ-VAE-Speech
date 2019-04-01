from experiments.experiment import Experiment
from experiments.checkpoint_utils import CheckpointUtils
from experiments.device_configuration import DeviceConfiguration
from error_handling.console_logger import ConsoleLogger

import json
import yaml
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from itertools import cycle


class Experiments(object):

    def __init__(self, experiments, seed):
        self._experiments = experiments
        self._seed = seed

    def run(self):
        Experiments.set_deterministic_on(self._seed)

        for experiment in self._experiments:
            experiment.run()
            torch.cuda.empty_cache() # Release the GPU memory cache

    def plot_losses(self, experiments_path):
        all_train_res_recon_errors = list()
        all_train_res_perplexities = list()
        all_results_paths = list()
        all_experiments_names = list()
        all_latest_epochs = list()

        for experiment in self._experiments:
            try:
                train_res_recon_errors, train_res_perplexities, latest_epoch = self._retreive_losses_values(experiments_path, experiment)
                all_train_res_recon_errors.append(train_res_recon_errors)
                all_train_res_perplexities.append(train_res_perplexities)
                all_results_paths.append(experiment.results_path)
                all_experiments_names.append(experiment.name)
                all_latest_epochs.append(latest_epoch)
            except:
                ConsoleLogger.error("Failed to retreive losses of experiment '{}'".format(experiment.name))

        self._plot_losses_figure(
            all_results_paths,
            all_experiments_names,
            all_train_res_recon_errors,
            all_train_res_perplexities,
            all_latest_epochs,
            merge_figures=False
        )

        self._plot_losses_figure(
            all_results_paths,
            all_experiments_names,
            all_train_res_recon_errors,
            all_train_res_perplexities,
            all_latest_epochs,
            merge_figures=True
        )

    def _retreive_losses_values(self, experiment_path, experiment):
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
            configuration = yaml.load(file)
        
        # Load the device configuration from the configuration state
        device_configuration = DeviceConfiguration.load_from_configuration(configuration)

        ConsoleLogger.status("Merge {} checkpoint losses of experiment '{}'".format(len(checkpoint_files), experiment_name))
        train_res_recon_errors, train_res_perplexities = CheckpointUtils.merge_experiment_losses(
            experiment_path,
            checkpoint_files,
            device_configuration
        )

        return train_res_recon_errors, train_res_perplexities, len(checkpoint_files)

    def _plot_losses_figure(self, all_results_paths, all_experiments_names, all_train_res_recon_errors,
        all_train_res_perplexities, all_latest_epochs, merge_figures,
        colormap_name='nipy_spectral'):

        def configure_ax1(ax, epochs, legend=False):
            ax.minorticks_off()
            ax.set_xticklabels(epochs)
            ax.grid(linestyle='--')
            ax.set_yscale('log')
            ax.set_title('Smoothed MSE')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            if legend:
                ax.legend()
            ax.grid(True)
            return ax

        def configure_ax2(ax, epochs, legend=False):
            ax.minorticks_off()
            ax.set_xticklabels(epochs)
            ax.grid(linestyle='--')
            ax.set_title('Smoothed Average codebook usage')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Perplexity')
            if legend:
                ax.legend()
            ax.grid(True)
            return ax

        linewidth = 2
        n_colors = len(all_train_res_recon_errors)
        colors = [plt.get_cmap(colormap_name)(1. * i/n_colors) for i in range(n_colors)]

        if merge_figures:
            latest_epoch = all_latest_epochs[0]
            for i in range(1, len(all_latest_epochs)):
                if all_latest_epochs[i] != latest_epoch:
                    raise ValueError('All experiments must have the same number of epochs to merge them')

            results_path = all_results_paths[0]
            experiment_name = 'merged_experiments'
            output_plot_path = results_path + os.sep + experiment_name + '.png'
            
            all_train_res_recon_error_smooth = list()
            all_train_res_perplexity_smooth = list()
            for i in range(len(all_train_res_recon_errors)):
                train_res_recon_error_smooth, train_res_perplexity_smooth = self._smooth_losses(
                    all_train_res_recon_errors[i],
                    all_train_res_perplexities[i]
                )
                all_train_res_recon_error_smooth.append(train_res_recon_error_smooth)
                all_train_res_perplexity_smooth.append(train_res_perplexity_smooth)

            epochs = range(1, latest_epoch + 1, 1)

            all_train_res_recon_error_smooth = np.asarray(all_train_res_recon_error_smooth)
            all_train_res_perplexity_smooth = np.asarray(all_train_res_perplexity_smooth)

            all_train_res_recon_error_smooth = np.reshape(all_train_res_recon_error_smooth, (n_colors, latest_epoch, all_train_res_recon_error_smooth.shape[1] // latest_epoch))
            all_train_res_perplexity_smooth = np.reshape(all_train_res_perplexity_smooth, (n_colors, latest_epoch, all_train_res_perplexity_smooth.shape[1] // latest_epoch))

            fig = plt.figure(figsize=(16, 8))

            ax = fig.add_subplot(1, 2, 1)
            for i in range(len(all_train_res_recon_error_smooth)):
                linecolor = colors[i] # TODO: compute a darker linecolor than facecolor
                facecolor = colors[i]
                mu = np.mean(all_train_res_recon_error_smooth[i], axis=1)
                sigma = np.std(all_train_res_recon_error_smooth[i], axis=1)
                t = np.arange(len(all_train_res_recon_error_smooth[i]))
                ax.plot(t, mu, linewidth=linewidth, label=all_experiments_names[i], c=linecolor)
                ax.fill_between(t, mu+sigma, mu-sigma, facecolor=facecolor, alpha=0.5)
            ax = configure_ax1(ax, epochs, legend=True)

            ax = fig.add_subplot(1, 2, 2)
            for i in range(len(all_train_res_perplexity_smooth)):
                linecolor = colors[i] # TODO: compute a darker linecolor than facecolor
                facecolor = colors[i]
                mu = np.mean(all_train_res_perplexity_smooth[i], axis=1)
                sigma = np.std(all_train_res_perplexity_smooth[i], axis=1)
                t = np.arange(len(all_train_res_perplexity_smooth[i]))
                ax.plot(t, mu, linewidth=linewidth, label=all_experiments_names[i], c=linecolor)
                ax.fill_between(t, mu+sigma, mu-sigma, facecolor=facecolor, alpha=0.5)
            ax = configure_ax2(ax, epochs, legend=True)

            fig.savefig(output_plot_path)
            plt.close(fig)

            ConsoleLogger.success("Saved figure at path '{}'".format(output_plot_path))
        else:
            for i in range(len(all_experiments_names)):
                results_path = all_results_paths[i]
                experiment_name = all_experiments_names[i]
                output_plot_path = results_path + os.sep + experiment_name + '.png'
                
                train_res_recon_error_smooth, train_res_perplexity_smooth = self._smooth_losses(
                    all_train_res_recon_errors[i],
                    all_train_res_perplexities[i]
                )

                latest_epoch = all_latest_epochs[i]
                linecolor = colors[i] # TODO: compute a darker linecolor than facecolor
                facecolor = colors[i]

                train_res_recon_error_smooth = np.asarray(train_res_recon_error_smooth)
                train_res_perplexity_smooth = np.asarray(train_res_perplexity_smooth)

                train_res_recon_error_smooth = np.reshape(train_res_recon_error_smooth, (latest_epoch, train_res_recon_error_smooth.shape[0] // latest_epoch))
                train_res_perplexity_smooth = np.reshape(train_res_perplexity_smooth, (latest_epoch, train_res_perplexity_smooth.shape[0] // latest_epoch))

                epochs = range(1, latest_epoch + 1, 1)

                fig = plt.figure(figsize=(16, 8))

                ax = fig.add_subplot(1, 2, 1)
                mu = np.mean(train_res_recon_error_smooth, axis=1)
                sigma = np.std(train_res_recon_error_smooth, axis=1)
                t = np.arange(len(train_res_recon_error_smooth))
                ax.plot(t, mu, linewidth=linewidth, label=all_experiments_names[i], c=linecolor)
                ax.fill_between(t, mu+sigma, mu-sigma, facecolor=facecolor, alpha=0.5)
                ax = configure_ax1(ax, epochs)

                ax = fig.add_subplot(1, 2, 2)
                mu = np.mean(train_res_perplexity_smooth, axis=1)
                sigma = np.std(train_res_perplexity_smooth, axis=1)
                t = np.arange(len(train_res_perplexity_smooth))
                ax.plot(t, mu, linewidth=linewidth, label=all_experiments_names[i], c=linecolor)
                ax.fill_between(t, mu+sigma, mu-sigma, facecolor=facecolor, alpha=0.5)
                ax = configure_ax2(ax, epochs)

                fig.savefig(output_plot_path)
                plt.close(fig)

                ConsoleLogger.success("Saved figure at path '{}'".format(output_plot_path))

    def _moving_average(self, a, n=10):
        """
        https://stackoverflow.com/a/14314054
        """
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def _smooth_losses(self, train_res_recon_errors, train_res_perplexities):
        maximum_window_length = 201
        train_res_recon_error_len = len(train_res_recon_errors)
        train_res_recon_error_len = train_res_recon_error_len if train_res_recon_error_len % 2 == 1 else train_res_recon_error_len - 1
        train_res_perplexity_len = len(train_res_perplexities)
        train_res_perplexity_len = train_res_perplexity_len if train_res_perplexity_len % 2 == 1 else train_res_perplexity_len - 1
        polyorder = 7

        train_res_recon_error_smooth = savgol_filter(
            train_res_recon_errors,
            maximum_window_length if train_res_recon_error_len > maximum_window_length else train_res_recon_error_len,
            polyorder
        )
        train_res_perplexity_smooth = savgol_filter(
            train_res_perplexities,
            maximum_window_length if train_res_perplexity_len > maximum_window_length else train_res_perplexity_len,
            polyorder
        )

        return train_res_recon_error_smooth, train_res_perplexity_smooth

    @staticmethod
    def set_deterministic_on(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def load(experiments_path):
        experiments = list()
        with open(experiments_path, 'r') as experiments_file:
            experiment_configurations = json.load(experiments_file)

            configuration = None
            with open(experiment_configurations['configuration_path'], 'r') as configuration_file:
                configuration = yaml.load(configuration_file)

            for experiment_configuration_key in experiment_configurations['experiments'].keys():
                experiment = Experiment(
                    name=experiment_configuration_key,
                    experiments_path=experiment_configurations['experiments_path'],
                    results_path=experiment_configurations['results_path'],
                    global_configuration=configuration,
                    experiment_configuration=experiment_configurations['experiments'][experiment_configuration_key]
                )
                experiments.append(experiment)
        
        return Experiments(experiments, experiment_configurations['seed'])
