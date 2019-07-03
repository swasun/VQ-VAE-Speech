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

from torch import nn


class GlobalConditioning(object):

    @staticmethod
    def compute(speaker_dic, speaker_ids, x_one_hot, device, gin_channels=128, expand=True):
        speakers_embedding = GlobalConditioning._Embedding(len(speaker_dic), gin_channels, padding_idx=None, std=0.1).to(device)

        # Extract the batch size and the signal length
        B, _, T = x_one_hot.size()

        # (B x 1) -> (B x 1 x gin_channels)
        global_conditioning = speakers_embedding(speaker_ids.view(B, -1).long())

        # (B x gin_channels x 1)
        global_conditioning = global_conditioning.transpose(1, 2)

        # Check if the result have the right dimension
        assert global_conditioning.dim() == 3

        """
        Return the global conditioning if the expand
        option is set to False
        """
        if not expand:
            return global_conditioning

        # Expand global conditioning features to all time steps
        expanded_global_conditioning = GlobalConditioning._expand_global_features(B, T, global_conditioning, bct=True)

        return expanded_global_conditioning

    @staticmethod
    def _Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
        m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        m.weight.data.normal_(0, std)
        return m

    @staticmethod
    def _expand_global_features(B, T, g, bct=True):
        """
        Expand global conditioning features to all time steps

        Args:
            B (int): Batch size.
            T (int): Time length.
            g (Tensor): Global features, (B x C) or (B x C x 1).
            bct (bool) : returns (B x C x T) if True, otherwise (B x T x C)

        Returns:
            Tensor: B x C x T or B x T x C or None
        """
        if g is None:
            return None
        g = g.unsqueeze(-1) if g.dim() == 2 else g
        if bct:
            g_bct = g.expand(B, -1, T)
            return g_bct.contiguous()
        else:
            g_btc = g.expand(B, -1, T).transpose(1, 2)
            return g_btc.contiguous()
