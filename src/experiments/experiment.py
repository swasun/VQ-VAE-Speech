from experiments.device_configuration import DeviceConfiguration
from experiments.model_factory import ModelFactory
from error_handling.console_logger import ConsoleLogger
from dataset.speech_dataset import SpeechDataset

import os


class Experiment(object):

    def __init__(self, results_path, global_configuration, experiment_configuration):
        self._results_path = results_path
        self._global_configuration = global_configuration
        self._experiment_configuration = experiment_configuration
        self._device_configuration = DeviceConfiguration.load_from_configuration(global_configuration)

        # Create a new configuration state from the default and the experiment specific aspects
        self._configuration = self._global_configuration
        for experiment_key in experiment_configuration.keys():
            if experiment_key in self._configuration:
                self._configuration[experiment_key] = experiment_configuration[experiment_key]

        # Set the result path and create the directory if it doesn't exist
        if not os.path.isdir(results_path):
            ConsoleLogger.status('Creating results directory at path: {}'.format(results_path))
            os.mkdir(results_path)
        else:
            ConsoleLogger.status('Results directory already created at path: {}'.format(results_path))

    def process(self):
        # Load the speech dataset
        ConsoleLogger.status('Loading speech dataset...')
        self._dataset = SpeechDataset(self._configuration, self._device_configuration.gpu_ids, self._device_configuration.use_cuda)

        # Build the model and the trainer from the configurations and the dataset
        self._model, self._trainer = ModelFactory.build(self._configuration, self._device_configuration, self._dataset)

        ConsoleLogger.status('Begins to train the model')
        self._trainer.train() # TODO: get the results
        #self._model.save(self._results_path + os.sep + 'model.pth') # Save our trained model
        #self._trainer.save_loss_plot(self._results_path + os.sep + 'losses.png') # Save the loss plot
