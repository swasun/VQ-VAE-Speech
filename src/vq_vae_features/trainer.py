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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import torch
import os
import pickle


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


class Trainer(object):

    def __init__(self, device, model, optimizer, data_stream, configuration, verbose=True):
        self._device = device
        self._model = model
        self._optimizer = optimizer
        self._data_stream = data_stream
        self._verbose = verbose
        self._configuration = configuration

    def train(self, experiments_path, experiment_name):
        self._model.train()

        ConsoleLogger.status('start epoch: {}'.format(self._configuration['start_epoch']))
        ConsoleLogger.status('num epoch: {}'.format(self._configuration['num_epochs']))

        for epoch in range(self._configuration['start_epoch'], self._configuration['num_epochs']):

            with tqdm(self._data_stream.training_loader) as train_bar:
                train_res_recon_error = []
                train_res_perplexity = []
                #named_parameters = []

                iteration = 0
                if self._configuration['record_codebook_stats']:
                    max_iterations_number = len(train_bar)
                    iterations_to_record = 10
                    iterations = list(np.arange(max_iterations_number, step=(max_iterations_number / iterations_to_record) - 1, dtype=int))

                for data in train_bar:
                    source = data['input_features'].to(self._device)
                    speaker_id = data['speaker_id'].to(self._device)
                    target = data['output_features'].to(self._device)

                    self._optimizer.zero_grad()

                    """
                    The perplexity a useful value to track during training.
                    It indicates how many codes are 'active' on average.
                    """
                    loss, _, perplexity, losses, encoding_indices, concatenated_quantized = \
                        self._model(source, target, self._data_stream.speaker_dic, speaker_id)

                    if self._configuration['record_codebook_stats'] and iteration in iterations:
                        embedding = self._model.vq.embedding.weight.data.cpu().detach().numpy()
                        codebook_stats_entry = {
                            'concatenated_quantized': concatenated_quantized.detach().cpu().numpy(),
                            'embedding': embedding,
                            'n_embedding': embedding.shape[0],
                            'encoding_indices': encoding_indices.detach().cpu().numpy(),
                            'speaker_ids': data['speaker_id'].to(self._device).detach().cpu().numpy(),
                            'batch_size': self._data_stream.training_batch_size
                        }
                        codebook_stats_entry_path = experiments_path + os.sep + experiment_name + '_' + str(epoch + 1) + '_' + str(iteration) + '_codebook-stats.pickle'
                        with open(codebook_stats_entry_path, 'wb') as file:
                            pickle.dump(codebook_stats_entry, file)

                    loss.backward()

                    self._optimizer.step()

                    perplexity_value = perplexity.item()
                    train_bar.set_description('Epoch {}: loss {:.4f} perplexity {:.3f}'.format(
                        epoch + 1, losses['loss'], perplexity_value))
                    
                    train_res_recon_error.append(losses)
                    train_res_perplexity.append(perplexity_value)

                    iteration += 1

                # FIXME

                #current_named_parameters = self._model.named_parameters()
                #named_parameters += current_named_parameters
                #print('current_named_parameters.shape: {}'.format(current_named_parameters.shape))
                #print('named_parameters.shape: {}'.format(named_parameters.shape))

                #plot_grad_flow(current_named_parameters)
                #plt.savefig('{}{}_{}.png'.format(experiments_path + os.sep, experiment_name, epoch + 1))

                torch.save({
                        'experiment_name': experiment_name,
                        'epoch': epoch + 1,
                        'model': self._model.state_dict(),
                        'optimizer': self._optimizer.state_dict(),
                        'train_res_recon_error': train_res_recon_error,
                        'train_res_perplexity': train_res_perplexity
                    },
                    os.path.join(experiments_path, '{}_{}_checkpoint.pth'.format(experiment_name, epoch + 1))
                )
