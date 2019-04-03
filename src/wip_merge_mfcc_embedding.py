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

import os
import matplotlib.pyplot as plt
import pylab
import numpy as np
import matplotlib.image as mpimg

if __name__ == "__main__":
    output_path = '.'
    imgs = [
        mpimg.imread('jitter30-ema_embedding_weight_mfcc_features.png'),
        mpimg.imread('baseline_embedding_weight_mfcc_features.png'),
        mpimg.imread('jitter12-ema_embedding_weight_mfcc_features.png')
    ]

    pylab.subplots_adjust(hspace=0.2)
    number_of_subplots = 3
    fig = plt.figure()
    for i, v in enumerate(range(number_of_subplots)):
        v = v + 1
        ax1 = pylab.subplot(number_of_subplots, 1, v)
        ax1.plot(imgs[i])
    fig.savefig(output_path + os.sep + '_merged_embedding_weight_cut.png')
    plt.close(fig)
