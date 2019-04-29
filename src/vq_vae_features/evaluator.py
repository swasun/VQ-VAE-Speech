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
from dataset.vctk_dataset import VCTKDataset
from vq_vae_speech.mu_law import MuLaw

import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import librosa
import librosa.output
import numpy as np
import umap
from textwrap import wrap


class Evaluator(object):

    def __init__(self, device, model, data_stream, configuration):
        self._device = device
        self._model = model
        self._data_stream = data_stream
        self._configuration = configuration

    def evaluate(self, results_path, experiment_name):
        self._reconstruct(results_path, experiment_name)
        self._compute_comparaison_plot(results_path, experiment_name)
        self._plot_quantized_embedding_spaces(results_path, experiment_name)
        #self._compute_wav(results_path, experiment_name)

    def _reconstruct(self, results_path, experiment_name):
        self._model.eval()

        (self._valid_originals, _, self._speaker_ids, self._target, self._wav_filename) = next(iter(self._data_stream.validation_loader))
        self._valid_originals = self._valid_originals.permute(0, 2, 1).contiguous().float()
        self._batch_size = self._valid_originals.size(0)
        self._target = self._target.permute(0, 2, 1).contiguous().float()
        self._wav_filename = self._wav_filename[0][0]
        self._valid_originals = self._valid_originals.to(self._device)
        self._target = self._target.to(self._device)

        z = self._model.encoder(self._valid_originals)
        z = self._model.pre_vq_conv(z)
        _, self._quantized, _, self._encodings, self._distances, self._encoding_indices, _ = self._model.vq(z)
        self._valid_reconstructions = self._model.decoder(self._quantized)[0]

    def _compute_comparaison_plot(self, results_path, experiment_name):
        def _load_wav(filename, sampling_rate, res_type, top_db):
            raw, original_rate = librosa.load(filename, sampling_rate, res_type=res_type)
            raw, original_rate = librosa.effects.trim(raw, top_db=top_db)
            raw /= np.abs(raw).max()
            raw = raw.astype(np.float32)
            return raw, original_rate

        audio, _ = _load_wav(self._wav_filename, self._configuration['sampling_rate'], self._configuration['res_type'], self._configuration['top_db'])
        preprocessed_audio = VCTKDataset.preprocessing_raw(audio, self._configuration['length'])
        spectrogram_parser = SpectrogramParser()
        spectrogram = spectrogram_parser.parse_audio(preprocessed_audio).contiguous()
        spectrogram = spectrogram.detach().cpu().numpy()

        valid_originals = self._valid_originals.detach().cpu()[0].numpy()

        probs = F.softmax(-self._distances[0], dim=1).detach().cpu().transpose(0, 1).contiguous()

        #target = self._target.detach().cpu()[0].numpy()

        valid_reconstructions = self._valid_reconstructions.detach().cpu().numpy()

        fig, axs = plt.subplots(5, 1, figsize=(30, 20), sharex=True)

        # Spectrogram of the original speech signal
        axs[0].set_title('Spectrogram of the original speech signal')
        self._plot_pcolormesh(spectrogram, fig, x=self._compute_unified_time_scale(spectrogram.shape[1]), axis=axs[0])

        # MFCC + d + a of the original speech signal
        axs[1].set_title('Augmented MFCC + d + a #filters=13+13+13 of the original speech signal')
        self._plot_pcolormesh(valid_originals, fig, x=self._compute_unified_time_scale(valid_originals.shape[1]), axis=axs[1])

        # Softmax of distances computed in VQ
        axs[2].set_title('Softmax of distances computed in VQ\n($||z_e(x) - e_i||^2_2$ with $z_e(x)$ the output of the encoder prior to quantization)')
        self._plot_pcolormesh(probs, fig, x=self._compute_unified_time_scale(probs.shape[1], downsampling_factor=2), axis=axs[2])

        encodings = self._encodings.detach().cpu().numpy()
        axs[3].set_title('Encodings')
        self._plot_pcolormesh(encodings[0].transpose(), fig, x=self._compute_unified_time_scale(encodings[0].transpose().shape[1],
            downsampling_factor=2), axis=axs[3])

        # Actual reconstruction
        axs[4].set_title('Actual reconstruction')
        self._plot_pcolormesh(valid_reconstructions, fig, x=self._compute_unified_time_scale(valid_reconstructions.shape[1]), axis=axs[4])

        output_path = results_path + os.sep + experiment_name + '_evaluation-comparaison-plot.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _compute_wav(self, results_path, experiment_name):
        output_mu = self._valid_reconstructions.argmax(dim=1).detach().cpu().float().numpy().squeeze()
        #output_mu = MuLaw.decode(output_mu)

        output_path = results_path + os.sep + experiment_name + '_output.wav'
        librosa.output.write_wav(output_path, output_mu, self._configuration['sampling_rate'])

    def _plot_pcolormesh(self, data, fig, x=None, y=None, axis=None):
        axis = plt.gca() if axis is None else axis # default axis if None
        x = np.arange(data.shape[1]) if x is None else x # default x shape if None
        y = np.arange(data.shape[0]) if y is None else y # default y shape if None
        c = axis.pcolormesh(x, y, data)
        fig.colorbar(c, ax=axis)

    def _plot_quantized_embedding_spaces(self, results_path, experiment_name):
        quantized = self._quantized.detach().cpu().numpy()
        concatenated_quantized = np.concatenate(quantized.transpose((2, 0, 1)))
        embedding = self._model.vq.embedding.weight.data.cpu().detach().numpy()
        n_embedding = embedding.shape[0]
        encoding_indices = self._encoding_indices.detach().cpu().numpy()
        encoding_indices = np.concatenate(encoding_indices)
        quantized_embedding_space = np.concatenate(
            (concatenated_quantized, embedding)
        )
        speaker_ids = self._speaker_ids.detach().cpu().numpy()
        time_speaker_ids = np.repeat(
            speaker_ids,
            quantized.shape[2],
            axis=1
        )
        time_speaker_ids = np.concatenate(time_speaker_ids)

        for n_neighbors in [3, 10, 50, 100]:
            map = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=0.0,
                metric='euclidean'
            )

            projection = map.fit_transform(quantized_embedding_space)

            self._plot_quantized_embedding_space(projection, n_neighbors, n_embedding,
                time_speaker_ids, encoding_indices, results_path, experiment_name)

    def _compute_unified_time_scale(self, shape, winstep=0.01, downsampling_factor=1):
        return np.arange(shape) * winstep * downsampling_factor

    def _plot_quantized_embedding_space(self, projection, n_neighbors, n_embedding,
        time_speaker_ids, encoding_indices, results_path, experiment_name, cmap='tab20',
        xlabel_width=60):

        def _configure_ax(ax, title=None, xlabel=None, ylabel=None, legend=False):
            ax.minorticks_off()
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if legend:
                ax.legend()
            ax.grid(True)
            ax.margins(x=0)
            return ax

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

        # Colored by speaker id
        axs[0].scatter(projection[:-n_embedding,0], projection[:-n_embedding, 1], s=10, c=time_speaker_ids, cmap=cmap) # audio frame colored by speaker id
        axs[0].scatter(projection[-n_embedding:,0], projection[-n_embedding:, 1], s=50, marker='x', c='k', alpha=0.8) # embedding
        xlabel = 'Embedding of ' + str(self._data_stream.validation_batch_size) + ' valuations' \
            ' with ' + str(n_neighbors) + ' neighbors with the audio frame points colored' \
            ' by speaker id and the embedding marks colored in black'
        axs[0] = _configure_ax(
            axs[0],
            xlabel='\n'.join(wrap(xlabel, xlabel_width))
        )

        # Colored by encoding indices
        axs[1].scatter(projection[:-n_embedding,0], projection[:-n_embedding, 1], s=10, c=encoding_indices, cmap=cmap) # audio frame colored by encoding indices
        axs[1].scatter(projection[-n_embedding:,0], projection[-n_embedding:, 1], s=50, marker='x', c=np.arange(n_embedding), cmap=cmap) # different color for each embedding
        xlabel = 'Embedding of ' + str(self._data_stream.validation_batch_size) + ' valuations' \
            ' with ' + str(n_neighbors) + ' neighbors with the audio frame points colored' \
            ' by encoding indices and the embedding marks colored by number of embedding' \
            ' vectors (using the same color map)'
        axs[1] = _configure_ax(
            axs[1],
            xlabel='\n'.join(wrap(xlabel, width=xlabel_width))
        )

        plt.suptitle('Quantized embedding space of ' + experiment_name, fontsize=16)

        output_path = results_path + os.sep + experiment_name + '_quantized_embedding_space-n' + str(n_neighbors) + '.png'
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
