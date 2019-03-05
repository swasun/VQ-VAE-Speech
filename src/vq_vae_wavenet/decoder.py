 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-WaveNet.                                              #
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

from residual_stack import ResidualStack
from wavenet_factory import WaveNetFactory
from conv1d_builder import Conv1DBuilder
from jitter import Jitter

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Decoder(nn.Module):
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, wavenet_type, use_kaiming_normal=False):
        super(Decoder, self).__init__()
        
        # Apply the randomized time-jitter regularization
        self._jitter = Jitter()
        
        """
        The jittered latent sequence is passed through a single
        convolutional layer with filter length 3 and 128 hidden
        units to mix information across neighboring timesteps.
        """
        self._conv_1 = Conv1DBuilder.build(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal
        )

        """
        The representation is then upsampled 320 times
        (to match the 16kHz audio sampling rate).
        """
        self._upsample = nn.Upsample(scale_factor=320, mode='nearest')

        self._wavenet = WaveNetFactory.build(wavenet_type)

    def forward(self, inputs):
        x, speaker_one_hot = inputs

        #if self._is_training and self._use_jitter:
        #    x = self._jitter(x)

        x = self._conv_1(x)

        upsampled = self._upsample(x)

        # Concatenate upsampled with speaker one-hot
        concatenated = np.concatenate((
                upsampled,
                speaker_one_hot
            ),
            axis=0
        )

        x = self._wavenet(concatenated)

        return x
