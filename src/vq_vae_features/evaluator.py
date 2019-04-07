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

from dataset.spectrogram_parser import SpectrogramParser
from vq_vae_speech.mu_law import MuLaw

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import librosa
import librosa.output
from python_speech_features.base import mel2hz
import numpy as np


class Evaluator(object):

    def __init__(self, device, model, data_stream, configuration):
        self._device = device
        self._model = model
        self._data_stream = data_stream
        self._configuration = configuration

    def evaluate(self, results_path, experiment_name):
        self._reconstruct(results_path, experiment_name)
        self._compute_comparaison_plot(results_path, experiment_name)
        #self._compute_embedding_plot(results_path, experiment_name)
        #self._compute_wav(results_path, experiment_name)
        #self._test(results_path, experiment_name)

    def _reconstruct(self, results_path, experiment_name):
        self._model.eval()

        (self._valid_originals, _, _, self._target, self._wav_filename) = next(iter(self._data_stream.validation_loader))
        self._valid_originals = self._valid_originals[0].view(1, self._valid_originals.size(1), self._valid_originals.size(2))
        self._target = self._target[0].view(1, self._target.size(1), self._target.size(2))
        self._wav_filename = self._wav_filename[0][0]
        self._valid_originals = self._valid_originals.to(self._device)
        self._target = self._target.to(self._device)

        z = self._model.encoder(self._valid_originals)
        z = self._model.pre_vq_conv(z)
        _, self._quantized, _, self._encodings, self._distances = self._model.vq(z)
        reconstructed_x = self._model.decoder(self._quantized)
        output_features_filters = self._configuration['output_features_filters'] * 3 if self._configuration['augment_output_features'] else self._configuration['output_features_filters']
        self._valid_reconstructions = reconstructed_x.view(-1, reconstructed_x.shape[1], output_features_filters)

    def _compute_comparaison_plot(self, results_path, experiment_name):
        spectrogram_parser = SpectrogramParser()
        spectrogram = spectrogram_parser.parse_audio(self._wav_filename).contiguous()
        spectrogram = spectrogram.detach().cpu().numpy()

        valid_originals = self._valid_originals.detach().cpu()[0].transpose(0, 1).numpy()

        probs = F.softmax(self._distances, dim=1).detach().cpu().transpose(0, 1).contiguous()

        target = self._target.detach().cpu()[0].transpose(0, 1).numpy()

        valid_reconstructions = self._valid_reconstructions.detach().cpu()[0].transpose(0, 1).numpy()

        _, axs = plt.subplots(5, 1, figsize=(30, 20))

        # Spectrogram of the original speech signal
        axs[0].set_title('Spectrogram of the original speech signal')
        axs[0].pcolor(spectrogram)

        # MFCC + d + a of the original speech signal
        axs[1].set_title('Augmented MFCC + d + a #filters=13+13+13 of the original speech signal')
        axs[1].pcolor(valid_originals)

        # logfbank of quantized target to reconstruct
        axs[2].set_title('logfbank of quantized target to reconstruct')
        axs[2].pcolor(target)

        # Softmax of distances computed in VQ
        axs[3].set_title('Softmax of distances computed in VQ\n($||z_e(x) - e_i||^2_2$ with $z_e(x)$ the output of the encoder prior to quantization)')
        axs[3].pcolor(probs)

        # Actual reconstruction
        axs[4].set_title('Actual reconstruction')
        axs[4].pcolor(valid_reconstructions)

        output_path = results_path + os.sep + experiment_name + '_evaluation-comparaison-plot.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _compute_embedding_plot(self, results_path, experiment_name):
        # Copyright (C) 2018 Zalando Research

        try:
            import umap
        except ImportError:
            raise ValueError('umap-learn not installed')

        map = umap.UMAP(
            n_neighbors=3,
            min_dist=0.1,
            metric='euclidean'
        )

        projection = map.fit_transform(self._model.vq.embedding.weight.data.cpu().detach().numpy())

        fig = plt.figure()
        plt.scatter(projection[:,0], projection[:,1], alpha=0.3)
        output_path = results_path + os.sep + experiment_name + '_evaluation-embedding-plot.png'
        fig.savefig(output_path)
        plt.close(fig)

    def _compute_wav(self, results_path, experiment_name):
        output_mu = self._valid_reconstructions.argmax(dim=1).detach().cpu().float().numpy().squeeze()
        #output_mu = MuLaw.decode(output_mu)

        output_path = results_path + os.sep + experiment_name + '_output.wav'
        librosa.output.write_wav(output_path, output_mu, self._configuration['sampling_rate'])

    def _test(self, results_path, experiment_name):
        avg_probs = torch.mean(self._encodings, dim=0)
        avg_probs = avg_probs.detach().cpu().numpy()

        encodings = self._encodings.detach().cpu().numpy()

        _, axs = plt.subplots(2, 1, figsize=(20, 20))

        axs[0].set_title('Encodings')
        axs[0].pcolor(encodings.transpose())

        axs[1].set_title('Average probs')
        axs[1].plot(avg_probs, np.arange(len(avg_probs)))

        output_path = results_path + os.sep + experiment_name + '_evaluation-test.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
