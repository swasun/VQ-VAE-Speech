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

        for i in range(len(self._experiments)):
            self._experiments[i].run()
            del self._experiments[i]
            torch.cuda.empty_cache() # Release the GPU memory cache

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
