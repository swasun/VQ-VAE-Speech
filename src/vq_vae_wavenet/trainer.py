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

from vq_vae_speech.mu_law import MuLaw

import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
from tqdm import tqdm
import librosa


class Trainer(object):

    def __init__(self, device, model, optimizer, data_stream, configuration, verbose=True):
        self._device = device
        self._model = model
        self._optimizer = optimizer
        self._data_stream = data_stream
        self._verbose = verbose
        self._train_res_recon_error = []
        self._train_res_perplexity = []
        self._configuration = configuration

    def train(self, experiments_path, experiment_name):
        for epoch in range(self._configuration['start_epoch'], self._configuration['num_epochs']):
            train_bar = tqdm(self._data_stream.training_loader)
            self._model.train()
            for data in train_bar:
                x_enc, x_dec, speaker_id, quantized, _ = data
                x_enc, x_dec, speaker_id, quantized = x_enc.to(self._device), x_dec.to(self._device), speaker_id.to(self._device), quantized.to(self._device)

                self._optimizer.zero_grad()
                loss, _, _ = self._model(x_enc, x_dec, speaker_id, quantized)
                loss.mean().backward()
                self._optimizer.step()
                #self._train_res_recon_error.append(recon_error.item())
                #self._train_res_perplexity.append(perplexity.item())

                train_bar.set_description('Epoch {}: loss {:.4f}'.format(epoch + 1, loss.mean().item()))

            self._model.eval()
            data_val = next(iter(self._data_stream.validation_loader))
            with torch.no_grad():
                x_enc_val, x_dec_val, speaker_id_val, quantized_val = data_val
                x_enc_val, x_dec_val, speaker_id_val, quantized_val = x_enc_val.to(self._device), x_dec_val.to(self._device), speaker_id_val.to(self._device), quantized_val.to(self._device)
                _, out = self._model(x_enc_val, x_dec_val, speaker_id_val, quantized_val)

                output = out.argmax(dim=1).detach().cpu().numpy().squeeze()
                input_mu = x_dec_val.argmax(dim=1).detach().cpu().numpy().squeeze()
                input = x_enc_val.detach().cpu().numpy().squeeze()

                output = MuLaw.decode(output)
                input_mu = MuLaw.decode(input_mu)

                #librosa.output.write_wav(os.path.join(save_path, '{}_output.wav'.format(epoch)), output, self._configuration['sampling_rate'])
                #librosa.output.write_wav(os.path.join(save_path, '{}_input_mu.wav'.format(epoch)), input_mu, self._configuration['sampling_rate'])
                #librosa.output.write_wav(os.path.join(save_path, '{}_input.wav'.format(epoch)), input, self._configuration['sampling_rate'])

            """torch.save({'epoch': epoch,
                        'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        'vq': vq.state_dict()
                        }, os.path.join(save_path, '{}_checkpoint.pth'.format(epoch)))"""
