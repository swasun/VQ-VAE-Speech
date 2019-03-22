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
from dataset.vctk_speech_stream import VCTKSpeechStream
from vq_vae_speech.speech_features import SpeechFeatures

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


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

    use_cuda = True
    device = 'cuda:0'
    gpu_ids = [0]
    
    data_stream = VCTKSpeechStream(configuration, gpu_ids, use_cuda)

    auto_encoder.eval()

    valid_originals, _, _, _ = next(iter(data_stream.training_loader))
    valid_originals = valid_originals.to(device)

    vq_output_eval = auto_encoder.pre_vq_conv(auto_encoder.encoder(valid_originals))
    _, valid_quantize, _, _ = auto_encoder.vq(vq_output_eval)
    valid_reconstructions = auto_encoder.decoder(valid_quantize)

    print('valid_reconstructions.size(): {}'.format(valid_reconstructions.size()))

    valid_reconstructions = valid_reconstructions.view(95, 13 * 3)
    print('valid_reconstructions.size() (reshaped): {}'.format(valid_reconstructions.size()))

    features = SpeechFeatures.features_from_name(
        name='mfcc',
        signal=valid_reconstructions.detach().cpu(),
        rate=16000,
        filters_number=13
    )
    print('features.shape: {}'.format(features.shape))
    features = np.swapaxes(features, 0, 1)
    print('features.shape (swapped): {}'.format(features.shape))
    plt.imshow(features)
    plt.show()
