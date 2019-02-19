# coding: utf-8

 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2017: Ryuichi Yamamoto.                                             #
 #                                                                                   #
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

from __future__ import with_statement, print_function, absolute_import


def wavenet(out_channels=256,
            layers=20,
            stacks=2,
            residual_channels=512,
            gate_channels=512,
            skip_out_channels=512,
            cin_channels=-1,
            gin_channels=-1,
            weight_normalization=True,
            dropout=1 - 0.95,
            kernel_size=3,
            n_speakers=None,
            upsample_conditional_features=False,
            upsample_scales=[16, 16],
            freq_axis_kernel_size=3,
            scalar_input=False,
            use_speaker_embedding=True,
            legacy=True,
            ):
    from wavenet_vocoder import WaveNet

    model = WaveNet(out_channels=out_channels, layers=layers, stacks=stacks,
                    residual_channels=residual_channels,
                    gate_channels=gate_channels,
                    skip_out_channels=skip_out_channels,
                    kernel_size=kernel_size, dropout=dropout,
                    weight_normalization=weight_normalization,
                    cin_channels=cin_channels, gin_channels=gin_channels,
                    n_speakers=n_speakers,
                    upsample_conditional_features=upsample_conditional_features,
                    upsample_scales=upsample_scales,
                    freq_axis_kernel_size=freq_axis_kernel_size,
                    scalar_input=scalar_input,
                    use_speaker_embedding=use_speaker_embedding,
                    legacy=legacy,
                    )

    return model
