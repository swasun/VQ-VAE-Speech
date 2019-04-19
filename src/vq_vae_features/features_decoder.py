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
 #   OUT OF OR IN CONNECTION+ WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from vq_vae_speech.residual_stack import ResidualStack
from vq_vae_speech.jitter import Jitter
from vq_vae_speech.conv1d_builder import Conv1DBuilder
from vq_vae_speech.conv_transpose1d_builder import ConvTranspose1DBuilder

import torch.nn as nn
import torch.nn.functional as F


class FeaturesDecoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens, use_kaiming_normal, use_jitter, jitter_probability):
        super(FeaturesDecoder, self).__init__()

        self._use_jitter = use_jitter

        if self._use_jitter:
            self._jitter = Jitter(jitter_probability)

        self._conv_1 = Conv1DBuilder.build(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
            use_kaiming_normal=use_kaiming_normal
        )

        self._upsample = nn.Upsample(scale_factor=2)
        
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal
        )
        
        self._conv_trans_1 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens, 
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
            use_kaiming_normal=use_kaiming_normal
        )

        self._conv_trans_2 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens, 
            out_channels=num_hiddens,
            kernel_size=3,
            padding=0,
            use_kaiming_normal=use_kaiming_normal
        )
        
        self._conv_trans_3 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=out_channels,
            kernel_size=2,
            padding=0,
            use_kaiming_normal=use_kaiming_normal
        )

    def forward(self, inputs):
        x = inputs

        if self._use_jitter and self.training:
            x = self._jitter(x)

        #print('x: {}'.format(x.size()))
        x = self._conv_1(x)
        #print('_conv_1: {}'.format(x.size()))

        x = self._upsample(x)
        #print('_upsample: {}'.format(x.size()))
        
        x = self._residual_stack(x)
        #print('_residual_stack: {}'.format(x.size()))
        
        x = F.relu(self._conv_trans_1(x))
        #print('_conv_trans_1: {}'.format(x.size()))

        x = F.relu(self._conv_trans_2(x))
        #print('_conv_trans_2: {}'.format(x.size()))

        x = self._conv_trans_3(x)
        #print('_conv_trans_3: {}'.format(x.size()))
        
        return x
