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

from vq_vae_wavenet.encoder import Encoder
from vq_vae_wavenet.decoder import Decoder
from vq_vae_wavenet.vector_quantizer import VectorQuantizer
from vq_vae_wavenet.vector_quantizer_ema import VectorQuantizerEMA
from vq_vae_wavenet.wavenet_factory import WaveNetFactory
from wavenet_vocoder import WaveNet

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechEncoder(nn.Module):
    def __init__(self, dim):  # dim == d
        super(SpeechEncoder, self).__init__()

        self.dim = dim
        self.conv1 = nn.Conv2d(1, dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.conv6 = nn.Conv2d(dim, dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x):

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        z = self.conv6(h)

        return z

class AutoEncoder(nn.Module):
    
    def __init__(self, wavenet_type, device, configuration, params, speaker_dic):
        super(AutoEncoder, self).__init__()
        
        """
        Create the Encoder with a fixed number of channel
        (3 as specified in the paper).
        """
        """self._encoder = Encoder(
            3,
            configuration.encoder_num_hiddens,
            configuration.encoder_num_residual_layers, 
            configuration.encoder_num_residual_hiddens,
            configuration.use_kaiming_normal
        )"""
        self._encoder = SpeechEncoder(512)

        self._pre_vq_conv = nn.Conv1d(
            #in_channels=configuration.encoder_num_hiddens, 
            #out_channels=configuration.embedding_dim,
            128,
            512,
            kernel_size=1, 
            stride=1
        )

        if configuration.decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(
                device,
                #configuration.num_embeddings,
                #configuration.embedding_dim, 
                128,
                512,
                configuration.commitment_cost,
                configuration.decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                device,
                #configuration.num_embeddings,
                #configuration.embedding_dim, 
                128,
                512,
                configuration.commitment_cost
            )

        """self._decoder = Decoder(
            3,
            configuration.decoder_num_hiddens,
            configuration.decoder_num_residual_layers, 
            configuration.decoder_num_residual_hiddens,
            wavenet_type,
            configuration.use_kaiming_normal
        )"""
        #self._decoder = WaveNetFactory.build(wavenet_type)
        self._decoder = WaveNet(params['quantize'],
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
                      upsample_scales=[2, 2, 2, 2, 2, 2]) # 64 downsamples

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

    def forward(self, x_enc, x_dec, global_condition, quantized_val):
        z = self._encoder(x_enc)

        #z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)

        local_condition = quantized
        local_condition = local_condition.squeeze(-1)
        x_dec = x_dec.squeeze(-1)
        x_recon = self._decoder(x_dec, local_condition, global_condition)
        x_recon = x_recon.unsqueeze(-1)

        return loss, x_recon, perplexity

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(self, path, decoder, device, configuration):
        model = AutoEncoder(decoder, device, configuration)
        model.load_state_dict(torch.load(path, map_location=device))
        return model
