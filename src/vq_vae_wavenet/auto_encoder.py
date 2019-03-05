 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 # Copyright (C) 2018 Zalando Research                                               #
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

from encoder import Encoder
from vector_quantizer import VectorQuantizer
from vector_quantizer_ema import VectorQuantizerEMA

import torch.nn as nn
import torch
import os


class AutoEncoder(nn.Module):
    
    def __init__(self, decoder, device, configuration):
        super(AutoEncoder, self).__init__()
        
        """
        Create the Encoder with a fixed number of channel
        (3 as specified in the paper).
        """
        self._encoder = Encoder(
            3,
            configuration.num_hiddens,
            configuration.num_residual_layers, 
            configuration.num_residual_hiddens,
            configuration.use_kaiming_normal
        )

        self._pre_vq_conv = nn.Conv2d(
            in_channels=configuration.num_hiddens, 
            out_channels=configuration.embedding_dim,
            kernel_size=1, 
            stride=1
        )

        if configuration.decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(
                device,
                configuration.num_embeddings,
                configuration.embedding_dim, 
                configuration.commitment_cost,
                configuration.decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                device,
                configuration.num_embeddings,
                configuration.embedding_dim,
                configuration.commitment_cost
            )

        self._decoder = decoder

    @property
    def vq_vae(self):
        return self._vq_vae

    @property
    def pre_vq_conv(self):
        return self._pre_vq_conv

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(self, path, decoder, device, configuration):
        model = AutoEncoder(decoder, device, configuration)
        model.load_state_dict(torch.load(path, map_location=device))
        return model

