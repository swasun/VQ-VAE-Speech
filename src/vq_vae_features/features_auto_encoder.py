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
from vq_vae_features.features_decoder import FeaturesDecoder
from vq.vector_quantizer import VectorQuantizer
from vq.vector_quantizer_ema import VectorQuantizerEMA
from error_handling.console_logger import ConsoleLogger

import torch.nn as nn
import torch
import torch.nn.functional as F


class FeaturesAutoEncoder(nn.Module):
    
    def __init__(self, configuration, device):
        super(FeaturesAutoEncoder, self).__init__()

        self._output_features_filters = configuration['output_features_filters'] * 3 if configuration['augment_output_features'] else configuration['output_features_filters']
        self._output_features_dim = configuration['output_features_dim']
        self._verbose = configuration['verbose']

        self._encoder = SpeechEncoder(
            in_channels=configuration['input_features_dim'],
            num_hiddens=configuration['num_hiddens'],
            num_residual_layers=configuration['num_residual_layers'],
            num_residual_hiddens=configuration['num_hiddens'],
            use_kaiming_normal=configuration['use_kaiming_normal'],
            input_features_type=configuration['input_features_type'],
            features_filters=configuration['input_features_filters'] * 3 if configuration['augment_input_features'] else configuration['input_features_filters'],
            sampling_rate=configuration['sampling_rate'],
            device=device,
            verbose=self._verbose
        )

        self._pre_vq_conv = nn.Conv1d(
            in_channels=configuration['num_hiddens'],
            out_channels=configuration['embedding_dim'],
            kernel_size=3,
            stride=1
        )

        if configuration['decay'] > 0.0:
            self._vq = VectorQuantizerEMA(
                num_embeddings=configuration['num_embeddings'],
                embedding_dim=configuration['embedding_dim'],
                commitment_cost=configuration['commitment_cost'],
                decay=configuration['decay'],
                device=device
            )
        else:
            self._vq = VectorQuantizer(
                num_embeddings=configuration['num_embeddings'],
                embedding_dim=configuration['embedding_dim'],
                commitment_cost=configuration['commitment_cost'],
                device=device
            )

        self._decoder = FeaturesDecoder(
            in_channels=configuration['embedding_dim'],
            out_channels=self._output_features_filters,
            num_hiddens=configuration['num_hiddens'],
            num_residual_layers=configuration['num_residual_layers'],
            num_residual_hiddens=configuration['residual_channels'],
            use_kaiming_normal=configuration['use_kaiming_normal'],
            use_jitter=configuration['use_jitter'],
            jitter_probability=configuration['jitter_probability'],
            verbose=self._verbose
        )

        self._device = device

        self._criterion = nn.MSELoss()
        #self._criterion = nn.KLDivLoss()

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

    def forward(self, x, y):
        x = x.permute(0, 2, 1).contiguous().float()
        y = y.permute(0, 2, 1).contiguous().float()

        z = self._encoder(x)
        if self._verbose:
            ConsoleLogger.status('[FEATURES_AE] _encoder output size: {}'.format(z.size()))

        z = self._pre_vq_conv(z)
        if self._verbose:
            ConsoleLogger.status('[FEATURES_AE] _pre_vq_conv output size: {}'.format(z.size()))

        vq_loss, quantized, perplexity, _, _ = self._vq(z)

        reconstructed_x = self._decoder(quantized)

        reconstructed_x = reconstructed_x.view(-1, self._output_features_filters, 47)
        
        reconstruction_loss = self._criterion(reconstructed_x, y.float()) # MSELoss

        #reconstruction_loss = self._criterion(F.log_softmax(reconstructed_x_features.view(1, self._output_features_filters, self._output_features_dim), dim=1), F.softmax(y.float(), dim=1)) # KLDivLoss

        loss = vq_loss + reconstruction_loss

        return loss, reconstructed_x, perplexity
