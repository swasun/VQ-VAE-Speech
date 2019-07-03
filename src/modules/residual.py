 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 # Copyright (C) 2018 Zalando Research                                               #
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

import torch.nn as nn


class Residual(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal):
        super(Residual, self).__init__()
        
        relu_1 = nn.ReLU(True)
        conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_residual_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        if use_kaiming_normal:
            conv_1 = nn.utils.weight_norm(conv_1)
            nn.init.kaiming_normal_(conv_1.weight)

        relu_2 = nn.ReLU(True)
        conv_2 = nn.Conv1d(
            in_channels=num_residual_hiddens,
            out_channels=num_hiddens,
            kernel_size=1,
            stride=1,
            bias=False
        )
        if use_kaiming_normal:
            conv_2 = nn.utils.weight_norm(conv_2)
            nn.init.kaiming_normal_(conv_2.weight)

        # All parameters same as specified in the paper
        self._block = nn.Sequential(
            relu_1,
            conv_1,
            relu_2,
            conv_2
        )
    
    def forward(self, x):
        return x + self._block(x)
