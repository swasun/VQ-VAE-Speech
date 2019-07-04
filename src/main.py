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
from dataset.vctk_speech_stream import VCTKSpeechStream
from dataset.vctk_features_stream import VCTKFeaturesStream
from experiments.pipeline_factory import PipelineFactory
from experiments.device_configuration import DeviceConfiguration
from experiments.experiments import Experiments
from evaluation.losses_plotter import LossesPlotter

import os
import argparse
import yaml
import sys
import json


def load_configuration(configuration_path):
    configuration = None
    with open(configuration_path, 'r') as configuration_file:
        configuration = yaml.load(configuration_file, Loader=yaml.FullLoader)
    return configuration

def update_configuration_from_experiments(experiments_configuration_path, configuration):
    experiment_configuration = None
    with open(experiments_configuration_path, 'r') as experiments_file:
        experiment_configuration = json.load(experiments_file)
    for entry in list(experiment_configuration.keys()):
        if entry == 'experiments':
            continue
        if entry in configuration:
            configuration[entry] = experiment_configuration[entry]
    return configuration


if __name__ == "__main__":
    default_experiments_configuration_path = '..' + os.sep + 'configurations' + os.sep + 'experiments_vq44-mfcc39.json'
    default_experiments_path = '..' + os.sep + 'experiments'
    default_configuration_path = '..' + os.sep + 'configurations' + os.sep + 'vctk_features.yaml'
    default_dataset_path = '..' + os.sep + 'data' + os.sep + 'vctk'
    default_results_path = '..' + os.sep + 'results'
    default_experiment_name = 'baseline'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--summary', nargs='?', default=None, type=str, help='The summary of the model based of a specified configuration file')
    parser.add_argument('--export_to_features', action='store_true', help='Export the VCTK dataset files to features')
    parser.add_argument('--compute_dataset_stats', action='store_true', help='Compute the mean and the std of the VCTK dataset')
    parser.add_argument('--experiments_configuration_path', nargs='?', default=default_experiments_configuration_path, type=str, help='The path of the experiments configuration file')
    parser.add_argument('--experiments_path', nargs='?', default=default_experiments_path, type=str, help='The path of the experiments ouput directory')
    parser.add_argument('--plot_experiments_losses', action='store_true', help='Plot the losses of the experiments based of the specified file in --experiments_configuration_path option')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--plot_comparaison_plot', action='store_true', help='Compute a comparaison plot for a single sample')
    parser.add_argument('--plot_quantized_embedding_spaces', action='store_true', help='Compute a 2D projection of the VQ codebook for a single sample')
    parser.add_argument('--compute_quantized_embedding_spaces_animation', action='store_true', help='Compute a 2D projection of the VQ codebook over training iterations')
    parser.add_argument('--plot_distances_histogram', action='store_true', help='Compute histograms of several distances to investiguate how close are the samples with the codebook')
    parser.add_argument('--compute_many_to_one_mapping', action='store_true', help='Compute the many to one mapping for all the samples')
    parser.add_argument('--compute_alignments', action='store_true', help='Compute the groundtruth alignments and those of the specified experiments')
    parser.add_argument('--compute_clustering_metrics', action='store_true', help='Compute the clustering metrics between the groundtruth and the empirical alignments')
    parser.add_argument('--compute_groundtruth_average_phonemes_number', action='store_true', help='Compute the average number of phonemes per groundtruth alignment')
    parser.add_argument('--plot_clustering_metrics_evolution', action='store_true', help='Compute the evolution of the clustering metrics accross different number of embedding vectors')
    parser.add_argument('--check_clustering_metrics_stability_over_seeds', action='store_true', help='Check the evolution of the clustering metrics statbility over different seed values')
    parser.add_argument('--plot_gradient_stats', action='store_true', help='Plot the gradient stats of the training')
    args = parser.parse_args()
    
    evaluation_options = {
        'plot_comparaison_plot': args.plot_comparaison_plot,
        'plot_quantized_embedding_spaces': args.plot_quantized_embedding_spaces,
        'compute_quantized_embedding_spaces_animation': args.compute_quantized_embedding_spaces_animation,
        'plot_distances_histogram': args.plot_distances_histogram,
        'compute_many_to_one_mapping': args.compute_many_to_one_mapping,
        'compute_alignments': args.compute_alignments,
        'compute_clustering_metrics': args.compute_clustering_metrics,
        'compute_groundtruth_average_phonemes_number': args.compute_groundtruth_average_phonemes_number,
        'plot_clustering_metrics_evolution': args.plot_clustering_metrics_evolution,
        'check_clustering_metrics_stability_over_seeds': args.check_clustering_metrics_stability_over_seeds,
        'plot_gradient_stats': args.plot_gradient_stats
    }

    # If specified, print the summary of the model using the CPU device
    if args.summary:
        configuration = load_configuration(args.summary)
        ConsoleLogger.status('Printing the summary of the model...')
        device_configuration = DeviceConfiguration.load_from_configuration(configuration)
        model = PipelineFactory.build(configuration, device_configuration,
            default_experiments_path, default_experiment_name, default_results_path)
        print(model)
        sys.exit(0)

    if args.plot_experiments_losses:
        LossesPlotter().plot_training_losses(
            Experiments.load(args.experiments_configuration_path).experiments,
            args.experiments_path
        )
        sys.exit(0)

    if args.export_to_features:
        configuration = load_configuration(default_configuration_path)
        update_configuration_from_experiments(args.experiments_configuration_path, configuration)
        device_configuration = DeviceConfiguration.load_from_configuration(configuration)
        data_stream = VCTKSpeechStream(configuration, device_configuration.gpu_ids, device_configuration.use_cuda)
        data_stream.export_to_features(default_dataset_path, configuration)
        ConsoleLogger.success("VCTK exported to a new features dataset at: '{}'".format(
            default_dataset_path + os.sep + configuration['features_path']))
        sys.exit(0)

    if args.evaluate:
        Experiments.load(args.experiments_configuration_path).evaluate(evaluation_options)
        ConsoleLogger.success('All evaluating experiments done')
        sys.exit(0)

    if args.compute_dataset_stats:
        configuration = load_configuration(default_configuration_path)
        update_configuration_from_experiments(args.experiments_configuration_path, configuration)
        device_configuration = DeviceConfiguration.load_from_configuration(configuration)
        data_stream = VCTKFeaturesStream(default_dataset_path, configuration, device_configuration.gpu_ids, device_configuration.use_cuda)
        data_stream.compute_dataset_stats()
        sys.exit(0)

    Experiments.load(args.experiments_configuration_path).train()
    ConsoleLogger.success('All training experiments done')
