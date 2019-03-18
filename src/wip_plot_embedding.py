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
from vq_vae_speech.speech_features import SpeechFeatures

import os
import yaml
import matplotlib.pyplot as plt
import pylab
import numpy as np


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def save_embedding_plot(embedding, path):
    try:
        import umap
    except ImportError:
        raise ValueError('umap-learn not installed')

    map = umap.UMAP(
        n_neighbors=3,
        min_dist=0.1,
        metric='euclidean'
    )

    projection = map.fit_transform(embedding.weight.data.cpu())

    fig = plt.figure()
    plt.scatter(projection[:,0], projection[:,1], alpha=0.3)
    fig.savefig(path)
    plt.close(fig)

def test_1(embedding):
    save_embedding_plot(
        embedding=embedding,
        path=results_path + os.sep + 'embedding.png'
    )

def test_2(embedding_weight):
    embedding_element = embedding_weight[0, :]
    plt.specgram(embedding_element, Fs=16000)
    plt.show()

def test_3(embedding_weight):
    plt.imshow(embedding_weight)

def test_4(embedding_weight):
    pylab.subplots_adjust(hspace=0.2)
    number_of_subplots = 3
    for i, v in enumerate(range(number_of_subplots)):
        v = v + 1
        ax1 = pylab.subplot(number_of_subplots, 1, v)
        ax1.specgram(embedding_weight[760+i, :], Fs=16000)
    plt.show()

def test_5(embedding):
    reversed_embedding_weight = embedding.weight.data.cpu().detach().permute(1, 0).contiguous().numpy()
    plt.imshow(reversed_embedding_weight)
    plt.colorbar()
    plt.show()

def test_6(embedding):
    features = SpeechFeatures.mfcc(
        signal=embedding.weight.data.detach().permute(1, 0).contiguous()[range(10), :],
        rate=16000,
        filters_number=13
    )
    features = np.swapaxes(features, 0, 1)
    plt.imshow(features)
    plt.show()


if __name__ == "__main__":
    # Dataset and model hyperparameters
    configuration = get_config('../configurations/vctk_features.yaml')

    results_path = '..' + os.sep + 'results'
    path = results_path + os.sep + 'loss_n1500_f13_jitter12_ema-80_kaming.pth'
    device = 'cuda:0'

    auto_encoder = FeaturesAutoEncoder.load(
        path=path,
        configuration=configuration,
        device=device
    ).to(device)

    embedding = auto_encoder.vq.embedding
    embedding_weight = embedding.weight.data.cpu().detach().numpy()
