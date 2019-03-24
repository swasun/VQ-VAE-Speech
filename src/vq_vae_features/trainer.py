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
from scipy.signal import savgol_filter
from tqdm import tqdm
import torch
import os

from vq_vae_speech.speech_features import SpeechFeatures


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
            train_bar = tqdm(self._data_stream.training_loader)
            train_res_recon_error = []
            train_res_perplexity = []

            for data in train_bar:
                (data, _, _, quantized) = data
                data = data.to(self._device)
                quantized = quantized.to(self._device)

                self._optimizer.zero_grad()

                """
                The perplexity a useful value to track during training.
                It indicates how many codes are 'active' on average.
                """
                loss, _, perplexity = self._model(data, quantized)
                loss.backward()

                self._optimizer.step()

                loss_value = loss.item()
                perplexity_value = perplexity.item()
                train_bar.set_description('Epoch {}: loss {:.4f} perplexity {:.3f}'.format(epoch + 1, loss_value, perplexity_value))
                
                train_res_recon_error.append(loss_value)
                train_res_perplexity.append(perplexity_value)

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
