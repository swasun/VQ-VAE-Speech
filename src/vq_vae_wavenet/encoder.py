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
from conv1d_builder import Conv1DBuilder

import torch.nn as nn
import torch.nn.functional as F
from python_speech_features.base import mfcc
from python_speech_features import delta
import numpy as np


class Encoder(nn.Module):
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, use_kaiming_normal=False):
        super(Encoder, self).__init__()

        """
        2 preprocessing convolution layers with filter length 3
        and residual connections.
        """

        self._conv_1 = Conv1DBuilder.build(
            in_channels=in_channels,
            out_channels=num_hiddens//2,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal
        )

        self._conv_2 = Conv1DBuilder.build(
            in_channels=num_hiddens//2,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal
        )

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_3 = Conv1DBuilder.build(
            in_channels=num_hiddens//2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            use_kaiming_normal=use_kaiming_normal
        )

        """
        2 convolutional layers with length 3 and
        residual connections.
        """

        self._conv_4 = Conv1DBuilder.build(
            in_channels=num_hiddens//2,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal
        )

        self._conv_5 = Conv1DBuilder.build(
            in_channels=num_hiddens//2,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal
        )

        """
        4 feedforward ReLu layers with residual connections.
        """

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal
        )

    def _compute_features_from_inputs(self, inputs):
        (rate, signal) = inputs
        mfcc_features = mfcc(signal, rate)
        d_mfcc_features = delta(mfcc_features, 2)
        a_mfcc_features = delta(d_mfcc_features, 2)
        concatenated_features = np.concatenate((
                mfcc_features,
                d_mfcc_features,
                a_mfcc_features
            ),
            axis=0
        )
        return concatenated_features

    def forward(self, inputs):
        features = self._compute_features_from_inputs(inputs)
        
        x = self._conv_1(features)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        x = F.relu(x)

        x = self._conv_4(x)
        x = F.relu(x)

        x = self._conv_5(x)
        x = F.relu(x)

        return self._residual_stack(x)
