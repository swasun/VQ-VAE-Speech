 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 # Copyright https://github.com/GwangsHong/VQVAE-pytorch                             #
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

import librosa
import numpy as np
import yaml
from python_speech_features.base import mfcc
from python_speech_features import delta


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def load_wav(filename, params):
    raw, _ = librosa.load(filename, params['sr'], res_type=params['res_type'])
    raw, _ = librosa.effects.trim(raw, params['top_db'])
    raw /= np.abs(raw).max()
    raw = raw.astype(np.float32)

    return raw

def mu_law_encode(x, mu = 256):
    x = x.astype(np.float32)
    y = np.sign(x) * np.log(1 + mu * np.abs(x)) / \
        np.log(1 + mu)
    y = np.digitize(y, 2 * np.arange(mu) / mu - 1) - 1
    return y.astype(np.long)

def mu_law_decode(y, mu = 256):
    y = y.astype(np.float32)
    y = 2 * y / mu - 1
    x = np.sign(y) / mu * ((mu) ** np.abs(y) - 1)
    return x.astype(np.float32)

def compute_mfcc_features(signal, rate=16000):
    mfcc_features = mfcc(signal, rate)
    d_mfcc_features = delta(mfcc_features, 2)
    a_mfcc_features = delta(d_mfcc_features, 2)
    concatenated_features = np.concatenate((
            mfcc_features,
            d_mfcc_features,
            a_mfcc_features
        ),
        axis=1
    )
    return concatenated_features
