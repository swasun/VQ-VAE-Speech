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

from vq_vae_wavenet.wavenet_encoder import WaveNetEncoder
from vq_vae_wavenet.wavenet_decoder import WaveNetDecoder
from vq_vae_wavenet.wavenet_factory import WaveNetFactory
from vq.vector_quantizer import VectorQuantizer
from vq.vector_quantizer_ema import VectorQuantizerEMA
from wavenet_vocoder import WaveNet

import torch
import torch.nn as nn


class WaveNetAutoEncoder(nn.Module):
    
    def __init__(self, wavenet_type, device, configuration, params, speaker_dic):
        super(WaveNetAutoEncoder, self).__init__()
        
        self._encoder = WaveNetEncoder(
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
            self._vq_vae = VectorQuantizerEMA(
                device,
                #configuration.num_embeddings,
                #configuration.embedding_dim, 
                params['k'],
                params['d'],
                configuration.commitment_cost,
                configuration.decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                device,
                #configuration.num_embeddings,
                #configuration.embedding_dim, 
                params['k'],
                params['d'],
                configuration.commitment_cost
            )

        self._decoder = WaveNetDecoder(
            params['k'],
            configuration.decoder_num_hiddens,
            configuration.decoder_num_residual_layers, 
            configuration.decoder_num_residual_hiddens,
            wavenet_type,
            configuration.use_kaiming_normal
        )
        #self._decoder = WaveNetFactory.build(wavenet_type)
        """self._decoder = WaveNet(params['quantize'],
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
                      upsample_scales=[2, 2, 2, 2, 2, 2]) # 64 downsamples"""

        self.criterion = nn.CrossEntropyLoss()
        self._device = device

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
        print('x_enc.size(): {}'.format(x_enc.size()))

        z = self._encoder(x_enc)
        print('z.size(): {}'.format(z.size()))

        z = self._pre_vq_conv(z)
        print('pre_vq_conv output size: {}'.format(z.size()))

        vq_loss, quantized, perplexity, _ = self._vq_vae(z)

        local_condition = quantized
        local_condition = local_condition.squeeze(-1)
        print('local_condition.size(): {}'.format(local_condition.size()))

        print('x_dec.size(): {}'.format(x_dec.size()))
        x_dec = x_dec.squeeze(-1)
        print('x_dec.size(): {}'.format(x_dec.size()))
        
        reconstructed_x = self._decoder(x_dec, local_condition, global_condition)
        reconstructed_x = reconstructed_x.unsqueeze(-1)

        reconstruction_loss = self.criterion(reconstructed_x, x_dec)

        loss = vq_loss + reconstruction_loss

        return loss, reconstructed_x, perplexity

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(self, path, decoder, device, configuration, params, speaker_dic):
        model = WaveNetAutoEncoder(decoder, device, configuration, params, speaker_dic)
        model.load_state_dict(torch.load(path, map_location=device))
        return model
