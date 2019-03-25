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
