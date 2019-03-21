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
from dataset.speech_dataset import SpeechDataset
from experiments.model_factory import ModelFactory
from experiments.device_configuration import DeviceConfiguration
from experiments.experiments import Experiments

import os
import argparse
import yaml
import sys


if __name__ == "__main__":
    default_experiments_configuration_path = '..' + os.sep + 'configurations' + os.sep + 'experiments_test.json'
    default_experiments_path = '..' + os.sep + 'experiments'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--summary', nargs='?', default=None, type=str, help='The summary of the model regarding a specified configuration file')
    parser.add_argument('--experiments_configuration_path', nargs='?', default=default_experiments_configuration_path, type=str, help='The path of the experiments configuration file')
    parser.add_argument('--experiments_path', nargs='?', default=default_experiments_path, type=str, help='The path of the experiments ouput directory')
    args = parser.parse_args()

    # If specified, print the summary of the model using the CPU device
    if args.summary:
        device = 'cpu'
        ConsoleLogger.status('Loading the configuration file {}...'.format(args.summary))
        configuration = None
        with open(args.summary, 'r') as configuration_file:
            configuration = yaml.load(configuration_file)
        ConsoleLogger.status('Printing the summary of the model...')
        device_configuration = DeviceConfiguration.load_from_configuration(configuration)
        dataset = SpeechDataset(configuration, device_configuration.gpu_ids, device_configuration.use_cuda)
        model = ModelFactory.build(configuration, device_configuration, dataset, with_trainer=False)
        print(model)
        sys.exit(0)

    Experiments.load(args.experiments_configuration_path).run()

    ConsoleLogger.success('Done.')
