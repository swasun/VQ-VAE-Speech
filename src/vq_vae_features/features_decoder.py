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
from vq_vae_speech.global_conditioning import GlobalConditioning
from error_handling.console_logger import ConsoleLogger

import torch.nn as nn
import torch.nn.functional as F


class FeaturesDecoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers,
        num_residual_hiddens, use_kaiming_normal, use_jitter, jitter_probability,
        verbose=False):

        super(FeaturesDecoder, self).__init__()

        self._use_jitter = use_jitter
        self._verbose = verbose

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
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] input size: {}'.format(x.size()))

        if self._use_jitter and self.training:
            x = self._jitter(x)

        # global conditioning here
        # speaker_embedding = GlobalConditioning.compute(speaker_dic, speaker_ids, x_one_hot, expand=True)
        # 22 x 64 + embedding speaker (40 au lieu de 128 par ex)

        x = self._conv_1(x)
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] _conv_1 output size: {}'.format(x.size()))

        x = self._upsample(x)
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] _upsample output size: {}'.format(x.size()))
        
        x = self._residual_stack(x)
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] _residual_stack output size: {}'.format(x.size()))
        
        x = F.relu(self._conv_trans_1(x))
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] _conv_trans_1 output size: {}'.format(x.size()))

        x = F.relu(self._conv_trans_2(x))
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] _conv_trans_2 output size: {}'.format(x.size()))

        x = self._conv_trans_3(x)
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] _conv_trans_3 output size: {}'.format(x.size()))
        
        return x
