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

from experiments.model_factory import ModelFactory
from experiments.device_configuration import DeviceConfiguration
from vq_vae_speech.speech_features import SpeechFeatures

import os
import matplotlib.pyplot as plt
import pylab
import numpy as np


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

    projection = map.fit_transform(embedding.weight.data.cpu().detach().numpy())

    fig = plt.figure()
    plt.scatter(projection[:,0], projection[:,1], alpha=0.3)
    fig.savefig(path)
    plt.close(fig)

def test_1(embedding, output_path):
    save_embedding_plot(
        embedding=embedding,
        path=output_path + os.sep + 'embedding.png'
    )

def test_2(embedding_weight, output_path, experiment_name):
    embedding_element = embedding_weight[0, :]
    fig = plt.figure()
    plt.specgram(embedding_element, Fs=16000)
    fig.savefig(output_path + os.sep + experiment_name + '_specgram.png')
    plt.close(fig)

def test_3(embedding_weight, output_path, experiment_name):
    plt.imsave(output_path + os.sep + experiment_name + '_embedding_weight.png', embedding_weight)

def test_4(embedding_weight, output_path, experiment_name):
    pylab.subplots_adjust(hspace=0.2)
    number_of_subplots = 3
    fig = plt.figure()
    for i, v in enumerate(range(number_of_subplots)):
        v = v + 1
        ax1 = pylab.subplot(number_of_subplots, 1, v)
        ax1.specgram(embedding_weight[760+i, :], Fs=16000)
    fig.savefig(output_path + os.sep + experiment_name + '_embedding_weight_cut.png')
    plt.close(fig)

def test_5(embedding, output_path, experiment_name):
    reversed_embedding_weight = embedding.weight.data.cpu().detach().permute(1, 0).contiguous().numpy()
    plt.plot(reversed_embedding_weight)
    plt.pcolor(reversed_embedding_weight)
    plt.colorbar()
    plt.savefig(output_path + os.sep + experiment_name + '_embedding_weight_colorbar.png')

def test_6(embedding, output_path, experiment_name):
    features = SpeechFeatures.mfcc(
        signal=embedding.weight.data.detach().permute(1, 0).contiguous(),
        rate=16000,
        filters_number=13
    )
    features = np.swapaxes(features, 0, 1)
    plt.imsave(output_path + os.sep + experiment_name + '_embedding_weight_mfcc_features.png', features)


if __name__ == "__main__":
    experiment_names = ['jitter12-ema', 'baseline', 'jitter30-ema']

    for experiment_name in experiment_names:
        model, _, configuration, data_stream = ModelFactory.load('../experiments', experiment_name)
        device_configuration = DeviceConfiguration.load_from_configuration(configuration)

        model.eval()

        embedding = model.vq.embedding
        embedding_weight = embedding.weight.data.cpu().detach().numpy()
        results_path = '..' + os.sep + 'results'
        
        #test_1(embedding, results_path)
        #test_2(embedding_weight, results_path, experiment_name)
        #test_3(embedding_weight, results_path, experiment_name)
        #test_4(embedding_weight, results_path, experiment_name)
        #test_5(embedding, results_path, experiment_name)
        test_6(embedding, results_path, experiment_name)
        