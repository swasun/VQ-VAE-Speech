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

from vq_vae_wavenet.wavenet_factory import WaveNetFactory
from vq_vae_speech.residual_stack import ResidualStack
from vq_vae_speech.conv1d_builder import Conv1DBuilder
from vq_vae_speech.jitter import Jitter
from wavenet_vocoder.wavenet import WaveNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WaveNetDecoder(nn.Module):
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, wavenet_type, params, speaker_dic, use_kaiming_normal=False):
        super(WaveNetDecoder, self).__init__()
        
        # Apply the randomized time-jitter regularization
        self._jitter = Jitter()
        
        """
        The jittered latent sequence is passed through a single
        convolutional layer with filter length 3 and 128 hidden
        units to mix information across neighboring timesteps.
        """
        self._conv_1 = Conv1DBuilder.build(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal
        )

        """
        The representation is then upsampled 320 times
        (to match the 16kHz audio sampling rate).
        """
        self._upsample = nn.Upsample(scale_factor=320, mode='nearest')

        #self._wavenet = WaveNetFactory.build(wavenet_type)
        self._wavenet = WaveNet(
            params['quantize'],
            params['n_layers'],
            params['n_loop'],
            params['residual_channels'],
            params['gate_channels'],
            params['skip_out_channels'],
            params['filter_size'],
            cin_channels=params['local_condition_dim'],
            gin_channels=params['global_condition_dim'],
            n_speakers=len(speaker_dic),
            upsample_conditional_features=True,
            upsample_scales=[2, 2, 2, 2, 2, 2] # 64 downsamples
        )

    def forward(self, x_dec, local_condition, global_condition):
        #if self._is_training and self._use_jitter:
        #    x = self._jitter(x)

        x = self._conv_1(torch.tensor(x_dec, dtype=torch.double)) # FIXME: improve this ugly fix
        #x = x_dec

        #upsampled = self._upsample(x)

        print('x_dec.size(): {}'.format(x.size()))
        print('x.size(): {}'.format(x.size()))
        #print('upsampled.size(): {}'.format(upsampled.size()))
        print('local_condition.size(): {}'.format(local_condition.size()))
        print('global_condition.size(): {}'.format(global_condition.size()))

        x = self._wavenet(x_dec, local_condition.double(), global_condition)

        return x
