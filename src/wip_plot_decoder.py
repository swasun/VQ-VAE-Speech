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
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model, _, configuration, data_stream = ModelFactory.load('../experiments', 'jitter12-ema')
    device_configuration = DeviceConfiguration.load_from_configuration(configuration)

    model.eval()

    valid_originals, _, _, _ = next(iter(data_stream.training_loader))
    valid_originals = valid_originals.to(device_configuration.device)

    vq_output_eval = model.pre_vq_conv(model.encoder(valid_originals))
    _, valid_quantize, _, _ = model.vq(vq_output_eval)
    valid_reconstructions = model.decoder(valid_quantize)

    valid_reconstructions = valid_reconstructions.view(95, 13 * 3)

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
