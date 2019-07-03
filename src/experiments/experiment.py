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
from experiments.pipeline_factory import PipelineFactory
from error_handling.console_logger import ConsoleLogger

import os
import yaml
import copy


class Experiment(object):

    def __init__(self, name, experiments_path, results_path, global_configuration,
        experiment_configuration, seed):

        self._name = name
        self._experiments_path = experiments_path
        self._results_path = results_path
        self._global_configuration = global_configuration
        self._experiment_configuration = experiment_configuration
        self._seed = seed

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
        configuration_file_already_exists = True if os.path.isfile(experiments_configuration_path) else False
        if not configuration_file_already_exists:
            self._device_configuration = DeviceConfiguration.load_from_configuration(global_configuration)

            # Create a new configuration state from the default and the experiment specific aspects
            self._configuration = copy.deepcopy(self._global_configuration)
            for experiment_key in experiment_configuration.keys():
                if experiment_key in self._configuration:
                    self._configuration[experiment_key] = experiment_configuration[experiment_key]

            # Save the configuration of the experiments
            with open(experiments_configuration_path, 'w') as file:
                yaml.dump(self._configuration, file)
        else:
            with open(experiments_configuration_path, 'r') as file:
                self._configuration = yaml.load(file, Loader=yaml.FullLoader)
                self._device_configuration = DeviceConfiguration.load_from_configuration(self._configuration)

        if configuration_file_already_exists:
            self._trainer, self._evaluator, self._configuration, self._device_configuration = \
                PipelineFactory.load(self._experiments_path, self._name, self._results_path)
        else:
            self._trainer, self._evaluator = PipelineFactory.build(self._configuration,
                self._device_configuration, self._experiments_path, self._name, self._results_path)

    @property
    def device_configuration(self):
        return self._device_configuration

    @property
    def experiment_path(self):
        return self._experiments_path

    @property
    def name(self):
        return self._name

    @property
    def seed(self):
        return self._seed

    @property
    def results_path(self):
        return self._results_path

    @property
    def configuration(self):
        return self._experiment_configuration

    def train(self):
        ConsoleLogger.status("Running the experiment called '{}'".format(self._name))
        ConsoleLogger.status('Begins to train the model')
        self._trainer.train()
        ConsoleLogger.success("Succeed to runned the experiment called '{}'".format(self._name))

    def evaluate(self, evaluation_options):
        ConsoleLogger.status("Running the experiment called '{}'".format(self._name))
        ConsoleLogger.status('Begins to evaluate the model')
        self._evaluator.evaluate(evaluation_options)
        ConsoleLogger.success("Succeed to runned the experiment called '{}'".format(self._name))
