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

from modules.residual_stack import ResidualStack
from modules.conv1d_builder import Conv1DBuilder
from modules.jitter import Jitter
from wavenet_vocoder.wavenet import WaveNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WaveNetDecoder(nn.Module):
    
    def __init__(self, configuration, device):
        super(WaveNetDecoder, self).__init__()

        self._use_jitter = configuration['use_jitter']
        
        # Apply the randomized time-jitter regularization
        if self._use_jitter:
            self._jitter = Jitter(configuration['jitter_probability'])
        
        """
        The jittered latent sequence is passed through a single
        convolutional layer with filter length 3 and 128 hidden
        units to mix information across neighboring timesteps.
        """
        self._conv_1 = Conv1DBuilder.build(
            in_channels=64,
            out_channels=768,
            kernel_size=2,
            use_kaiming_normal=configuration['use_kaiming_normal']
        )

        #self._wavenet = WaveNetFactory.build(wavenet_type)
        self._wavenet = WaveNet(
            configuration['quantize'],
            configuration['n_layers'],
            configuration['n_loop'],
            configuration['residual_channels'],
            configuration['gate_channels'],
            configuration['skip_out_channels'],
            configuration['filter_size'],
            cin_channels=configuration['local_condition_dim'],
            gin_channels=configuration['global_condition_dim'],
            n_speakers=configuration['speaker_dic_len'],
            upsample_conditional_features=True,
            upsample_scales=[2, 2, 2, 2, 2, 12] # 768
            #upsample_scales=[2, 2, 2, 2, 12]
        )

        self._device = device

    def forward(self, y, local_condition, global_condition):
        if self._use_jitter and self.training:
            local_condition = self._jitter(local_condition)

        local_condition = self._conv_1(local_condition)

        x = self._wavenet(y, local_condition, global_condition)

        return x
