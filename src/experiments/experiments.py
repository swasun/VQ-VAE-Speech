from experiments.experiment import Experiment

import json
import yaml
import torch
import numpy as np
import random


class Experiments(object):

    def __init__(self, experiments, seed):
        self._experiments = experiments
        self._seed = seed

    def run(self):
        Experiments.set_deterministic_on(self._seed)

        for experiment in self._experiments:
            experiment.process()

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
                    results_path=experiment_configurations['results_path'],
                    global_configuration=configuration,
                    experiment_configuration=experiment_configurations['experiments'][experiment_configuration_key]
                )
                experiments.append(experiment)
        
        return Experiments(experiments, experiment_configurations['seed'])
