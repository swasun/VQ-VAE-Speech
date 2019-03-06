 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-WaveNet.                                               #
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

from vq_vae_wavenet.auto_encoder import AutoEncoder
from vq_vae_wavenet.trainer import Trainer
from vq_vae_wavenet.evaluator import Evaluator
from vq_vae_wavenet.configuration import Configuration
from vq_vae_wavenet.wavenet_type import WaveNetType

import torch
import torch.optim as optim
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', nargs='?', default=Configuration.default_batch_size, type=int, help='The size of the batch during training')
    parser.add_argument('--num_training_updates', nargs='?', default=Configuration.default_num_training_updates, type=int, help='The number of updates during training')
    parser.add_argument('--encoder_num_hiddens', nargs='?', default=Configuration.default_encoder_num_hiddens, type=int, help='The number of hidden neurons in each layer of the Encoder')
    parser.add_argument('--encoder_num_residual_hiddens', nargs='?', default=Configuration.default_encoder_num_residual_hiddens, type=int, help='The number of hidden neurons in each layer within a residual block of the Encoder')
    parser.add_argument('--encoder_num_residual_layers', nargs='?', default=Configuration.default_encoder_num_residual_layers, type=int, help='The number of residual layers in a residual stack of the Encoder')
    parser.add_argument('--decoder_num_hiddens', nargs='?', default=Configuration.default_decoder_num_hiddens, type=int, help='The number of hidden neurons in each layer of the Decoder')
    parser.add_argument('--decoder_num_residual_hiddens', nargs='?', default=Configuration.default_decoder_num_residual_hiddens, type=int, help='The number of hidden neurons in each layer within a residual block of the Decoder')
    parser.add_argument('--decoder_num_residual_layers', nargs='?', default=Configuration.default_decoder_num_residual_layers, type=int, help='The number of residual layers in a residual stack of the Decoder')
    parser.add_argument('--embedding_dim', nargs='?', default=Configuration.default_embedding_dim, type=int, help='Representing the dimensionality of the tensors in the quantized space')
    parser.add_argument('--num_embeddings', nargs='?', default=Configuration.default_num_embeddings, type=int, help='The number of vectors in the quantized space')
    parser.add_argument('--commitment_cost', nargs='?', default=Configuration.default_commitment_cost, type=float, help='Controls the weighting of the loss terms')
    parser.add_argument('--decay', nargs='?', default=Configuration.default_decay, type=float, help='Decay for the moving averages (set to 0.0 to not use EMA)')
    parser.add_argument('--learning_rate', nargs='?', default=Configuration.default_learning_rate, type=float, help='The learning rate of the optimizer during training updates')
    parser.add_argument('--use_kaiming_normal', nargs='?', default=Configuration.default_use_kaiming_normal, type=bool, help='Use the weight normalization proposed in [He, K et al., 2015]')
    parser.add_argument('--unshuffle_dataset', default=not Configuration.default_shuffle_dataset, action='store_true', help='Do not shuffle the dataset before training')
    parser.add_argument('--data_path', nargs='?', default='data', type=str, help='The path of the data directory')
    parser.add_argument('--results_path', nargs='?', default='results', type=str, help='The path of the results directory')
    parser.add_argument('--loss_plot_name', nargs='?', default='loss.png', type=str, help='The file name of the training loss plot')
    parser.add_argument('--model_name', nargs='?', default='model.pth', type=str, help='The file name of trained model')
    args = parser.parse_args()

    # Dataset and model hyperparameters
    configuration = Configuration.build_from_args(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if cuda is available

    # Set the result path and create the directory if it doesn't exist
    results_path = '..' + os.sep + args.results_path
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    
    dataset_path = '..' + os.sep + args.data_path

    auto_encoder = AutoEncoder(WaveNetType.WaveNet, device, configuration).to(device) # Create an AutoEncoder model using our GPU device

    print(auto_encoder)

    """optimizer = optim.Adam(auto_encoder.parameters(), lr=configuration.learning_rate, amsgrad=True) # Create an Adam optimizer instance
    trainer = Trainer(device, auto_encoder, optimizer, dataset) # Create a trainer instance
    trainer.train(configuration.num_training_updates) # Train our model on the CIFAR10 dataset
    auto_encoder.save(results_path + os.sep + args.model_name) # Save our trained model
    trainer.save_loss_plot(results_path + os.sep + args.loss_plot_name) # Save the loss plot

    evaluator = Evaluator(device, auto_encoder, dataset) # Create en Evaluator instance to evaluate our trained model"""
    