 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-Speech.                                              #
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

from vq_vae_speech.residual_stack import ResidualStack
from vq_vae_speech.conv1d_builder import Conv1DBuilder
from vq_vae_speech.utils import compute_features_from_inputs

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpeechEncoder(nn.Module):
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, device, use_kaiming_normal=False):
        super(SpeechEncoder, self).__init__()

        self._device = device

        """
        2 preprocessing convolution layers with filter length 3
        and residual connections.
        """

        self._conv_1 = Conv1DBuilder.build(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal
        )

        self._conv_2 = Conv1DBuilder.build(
            in_channels=num_hiddens,
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
            in_channels=num_hiddens,
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
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal
        )

        self._conv_5 = Conv1DBuilder.build(
            in_channels=num_hiddens,
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

    def forward(self, inputs):
        features = compute_features_from_inputs(inputs)
        features_tensor = torch.tensor(features).view(-1, 95, 39).to(self._device)
        print('inputs.size(): {}'.format(inputs.size()))
        print('features.shape: {}'.format(features.shape))
        print('features_tensor.size(): {}'.format(features_tensor.size()))

        #features_tensor = inputs

        x = self._conv_1(features_tensor)
        print('conv_1 output size: {}'.format(x.size()))
        x = F.relu(x)
        
        x = self._conv_2(x)
        print('conv_2 output size: {}'.format(x.size()))
        x = F.relu(x)
        
        x = self._conv_3(x)
        print('conv_3 output size: {}'.format(x.size()))
        x = F.relu(x)

        x = self._conv_4(x)
        print('conv_4 output size: {}'.format(x.size()))
        x = F.relu(x)

        x = self._conv_5(x)
        print('conv_5 output size: {}'.format(x.size()))
        x = F.relu(x)

        x = self._residual_stack(x)
        print('residual_stack output size: {}'.format(x.size()))

        return x
