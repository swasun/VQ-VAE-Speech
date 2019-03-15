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

from vq_vae_features.features_auto_encoder import FeaturesAutoEncoder
from vq_vae_features.trainer import Trainer
from vq_vae_features.evaluator import Evaluator
from dataset.speech_dataset import SpeechDataset

import torch
import torch.optim as optim
import os
import argparse
import yaml


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', nargs='?', default='data', type=str, help='The path of the data directory')
    parser.add_argument('--results_path', nargs='?', default='results', type=str, help='The path of the results directory')
    parser.add_argument('--loss_plot_name', nargs='?', default='loss.png', type=str, help='The file name of the training loss plot')
    parser.add_argument('--model_name', nargs='?', default='model.pth', type=str, help='The file name of trained model')
    parser.add_argument('--original_images_name', nargs='?', default='original_images.png', type=str, help='The file name of the original images used in evaluation')
    parser.add_argument('--validation_images_name', nargs='?', default='validation_images.png', type=str, help='The file name of the reconstructed images used in evaluation')
    args = parser.parse_args()

    # Dataset and model hyperparameters
    configuration = get_config('../configurations/vctk.yaml')

    #use_cuda = torch.cuda.is_available()
    #device = torch.device('cuda' if use_cuda else 'cpu') # Use GPU if cuda is available
    #gpu_ids = [i for i in range(torch.cuda.device_count())]
    use_cuda = True
    device = 'cuda:0'
    gpu_ids = [0]

    # Set the result path and create the directory if it doesn't exist
    results_path = '..' + os.sep + args.results_path
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    
    dataset = SpeechDataset(configuration, gpu_ids, use_cuda)

    auto_encoder = FeaturesAutoEncoder(configuration, device).to(device) # Create an AutoEncoder model using our GPU device

    optimizer = optim.Adam(auto_encoder.parameters(), lr=configuration['learning_rate'], amsgrad=True) # Create an Adam optimizer instance
    trainer = Trainer(device, auto_encoder, optimizer, dataset) # Create a trainer instance
    trainer.train(configuration['num_epochs'])
    auto_encoder.save(results_path + os.sep + args.model_name) # Save our trained model
    trainer.save_loss_plot(results_path + os.sep + args.loss_plot_name) # Save the loss plot

    #evaluator = Evaluator(device, auto_encoder, dataset) # Create en Evaluator instance to evaluate our trained model
    #evaluator.reconstruct() # Reconstruct our images from the embedded space
    #evaluator.save_original_images_plot(results_path + os.sep + args.original_images_name) # Save the original images for comparaison purpose
