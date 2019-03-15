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

from vq_vae_speech.speech_encoder import SpeechEncoder
from vq_vae_speech.speech_features import SpeechFeatures
from vq_vae_features.features_decoder import FeaturesDecoder
from vq.vector_quantizer import VectorQuantizer
from vq.vector_quantizer_ema import VectorQuantizerEMA

import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import numpy as np


class FeaturesAutoEncoder(nn.Module):
    
    def __init__(self, device, configuration, params):
        super(FeaturesAutoEncoder, self).__init__()
        
        self._encoder = SpeechEncoder(
            in_channels=95,
            num_hiddens=params['d'],
            num_residual_layers=2,
            num_residual_hiddens=params['d'],
            device=device,
            use_kaiming_normal=configuration.use_kaiming_normal
        )

        self._pre_vq_conv = nn.Conv1d(
            #in_channels=configuration.encoder_num_hiddens, 
            #out_channels=configuration.embedding_dim,
            768,
            64,
            kernel_size=1,
            stride=1
        )

        if configuration.decay > 0.0:
            self._vq = VectorQuantizerEMA(
                device,
                #configuration.num_embeddings,
                #configuration.embedding_dim, 
                params['k'],
                params['d'],
                configuration.commitment_cost,
                configuration.decay
            )
        else:
            self._vq = VectorQuantizer(
                device,
                #configuration.num_embeddings,
                #configuration.embedding_dim, 
                params['k'],
                params['d'],
                configuration.commitment_cost
            )

        self._decoder = FeaturesDecoder(
            params['k'],
            params['d'],
            2,
            params['d'],
            configuration.use_kaiming_normal
        )

        self.criterion = nn.MSELoss()
        self._device = device

    @property
    def vq(self):
        return self._vq

    @property
    def pre_vq_conv(self):
        return self._pre_vq_conv

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def forward(self, x_enc, target):
        z = self._encoder(x_enc)
        #print('z.size(): {}'.format(z.size()))

        z = self._pre_vq_conv(z)
        #print('pre_vq_conv output size: {}'.format(z.size()))

        vq_loss, quantized, perplexity, _ = self._vq(z)

        reconstructed_x = self._decoder(quantized)
        reconstructed_x = reconstructed_x.view(95, 39)
        #print('reconstructed_x.size() reshaped: {}'.format(reconstructed_x.size()))

        #target_features = SpeechFeatures.mfcc(target)
        target_features = SpeechFeatures.logfbank(target)
        tensor_target_features = torch.tensor(target_features).to(self._device)
        """print()
        print('reconstructed_x.size(): {}'.format(reconstructed_x.size()))
        print('target.size(): {}'.format(target.size()))
        print('target_features.shape: {}'.format(target_features.shape))
        print('tensor_target_features.size(): {}'.format(tensor_target_features.size()))"""
        reconstruction_loss = self.criterion(reconstructed_x, tensor_target_features)
        loss = vq_loss + reconstruction_loss

        return loss, reconstructed_x, perplexity

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(self, path, configuration, device, params):
        model = FeaturesAutoEncoder(device, configuration, params)
        model.load_state_dict(torch.load(path, map_location=device))
        return model
