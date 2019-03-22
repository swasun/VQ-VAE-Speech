from experiments.device_configuration import DeviceConfiguration
from experiments.model_factory import ModelFactory
from error_handling.console_logger import ConsoleLogger
from dataset.vctk_features_stream import VCTKFeaturesStream

import os
import yaml


class Experiment(object):

    def __init__(self, name, experiments_path, results_path, global_configuration, experiment_configuration):
        self._name = name
        self._experiments_path = experiments_path
        self._results_path = results_path
        self._global_configuration = global_configuration
        self._experiment_configuration = experiment_configuration

        # Create the results path directory if it doesn't exist
        if not os.path.isdir(results_path):
            ConsoleLogger.status('Creating results directory at path: {}'.format(results_path))
            os.mkdir(results_path)
        else:
            ConsoleLogger.status('Results directory already created at path: {}'.format(results_path))

        # Create the experiments path directory if it doesn't exist
        if not os.path.isdir(experiments_path):
            ConsoleLogger.status('Creating experiments directory at path: {}'.format(experiments_path))
            os.mkdir(experiments_path)
        else:
            ConsoleLogger.status('Experiments directory already created at path: {}'.format(experiments_path))

        experiments_configuration_path = experiments_path + os.sep + name + '_configuration.yaml'
        self._configuration_file_already_exists = True if os.path.isfile(experiments_configuration_path) else False
        if not self._configuration_file_already_exists:
            self._device_configuration = DeviceConfiguration.load_from_configuration(global_configuration)

            # Create a new configuration state from the default and the experiment specific aspects
            self._configuration = self._global_configuration
            for experiment_key in experiment_configuration.keys():
                if experiment_key in self._configuration:
                    self._configuration[experiment_key] = experiment_configuration[experiment_key]

            # Save the configuration of the experiments
            with open(experiments_configuration_path, 'w') as file:
                yaml.dump(self._configuration, file)
        else:
            with open(experiments_configuration_path, 'r') as file:
                self._configuration = yaml.load(file)
                self._device_configuration = DeviceConfiguration.load_from_configuration(self._configuration)

    @property
    def device_configuration(self):
        return self._device_configuration

    def run(self):
        ConsoleLogger.status("Running the experiment called '{}'".format(self._name))

        def create_from_scratch(configuration, device_configuration):
            # Load the data stream
            ConsoleLogger.status('Loading data stream')
            data_stream = VCTKFeaturesStream('../data/vctk', configuration, device_configuration.gpu_ids, device_configuration.use_cuda)

            # Build the model and the trainer from the configurations and the data stream
            model, trainer = ModelFactory.build(configuration, device_configuration, data_stream)

            return model, trainer, data_stream, configuration

        if self._configuration_file_already_exists:
            ConsoleLogger.status('Configuration file already exists. Loading...')
            try:
                self._model, self._trainer, _, self._data_stream = ModelFactory.load(self._experiments_path, self._name)
            except:
                ConsoleLogger.error('Failed to load existing configuration. Building a new model...')
                self._model, self._trainer, self._data_stream, self._configuration = create_from_scratch(self._configuration, self._device_configuration)
        else:
            self._model, self._trainer, self._data_stream, self._configuration = create_from_scratch(self._configuration, self._device_configuration)

        ConsoleLogger.status('Begins to train the model')
        #self._trainer.train(self._experiments_path, self._name)

        ConsoleLogger.success("Succeed to runned the experiment called '{}'".format(self._name))
