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

import matplotlib.pyplot as plt
import torch.nn.functional as F


class Evaluator(object):

    def __init__(self, device, model, data_stream):
        self._device = device
        self._model = model
        self._data_stream = data_stream

    def evaluate(self, experiments_path, experiment_name):
        self._reconstruct()
        self._plot()

    def _reconstruct(self):
        self._model.eval()

        (self._valid_originals, _, _, self._target, self._wav_filename) = next(iter(self._data_stream.validation_loader))
        self._valid_originals = self._valid_originals[0].view(1, self._valid_originals.size(1), self._valid_originals.size(2))
        self._target = self._target[0].view(1, self._target.size(1), self._target.size(2))
        self._wav_filename = self._wav_filename[0][0]
        self._valid_originals = self._valid_originals.to(self._device)
        self._target = self._target.to(self._device)

        z = self._model.encoder(self._valid_originals)
        z = self._model.pre_vq_conv(z)
        _, self._quantized, _, _, self._distances = self._model.vq(z)
        reconstructed_x = self._model.decoder(self._quantized)
        self._valid_reconstructions = reconstructed_x.view(-1, reconstructed_x.shape[1], 13 * 3)

    def _plot(self):
        spectrogram_parser = SpectrogramParser()
        spectrogram = spectrogram_parser.parse_audio(self._wav_filename).contiguous()
        spectrogram = spectrogram.detach().cpu().numpy()

        probs = F.softmax(self._distances).detach().cpu().transpose(0, 1).contiguous()

        target = self._target.detach().cpu()[0].transpose(0, 1).numpy()

        valid_reconstructions = self._valid_reconstructions.detach().cpu()[0].transpose(0, 1).numpy()

        _, axs = plt.subplots(4, 1, figsize=(20, 20))
        axs[0].pcolor(spectrogram) # Spectrogram of the original speech signal
        axs[1].pcolor(target) # logfbank of quantized target to reconstruct
        axs[2].pcolor(probs) # Softmax of distances computed in VQ
        axs[3].pcolor(valid_reconstructions) # Actual reconstruction
        plt.savefig('test_evaluation.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    def _save_embedding_plot(self, path):
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

        projection = map.fit_transform(self._model.vq.embedding.weight.data.cpu())

        fig = plt.figure()
        plt.scatter(projection[:,0], projection[:,1], alpha=0.3)
        fig.savefig(path)
        plt.close(fig)
