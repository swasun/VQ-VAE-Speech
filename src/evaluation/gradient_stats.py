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

from error_handling.console_logger import ConsoleLogger

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from operator import itemgetter
from tqdm import trange


class GradientStats(object):

    @staticmethod
    def build_gradient_entry(named_parameters):
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().detach().cpu().numpy())
                max_grads.append(p.grad.abs().max().detach().cpu().numpy())
        return {
            'ave_grads': ave_grads,
            'max_grads': max_grads,
            'layers': layers
        }

    @staticmethod
    def plot_gradient_flow(named_parameters, ax, set_xticks=False, set_ylabels=False, set_title=False):
        """
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        @source Inspired of https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10 by Roshan Rane
        """

        ave_grads = named_parameters['ave_grads']
        max_grads = named_parameters['max_grads']
        layers = named_parameters['layers']
        layers = [layer.replace('weight', '').replace('layer', '').replace('_', '').replace('.', ' ') for layer in layers]

        ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color='c')
        ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color='b')
        ax.hlines(0, 0, len(ave_grads) + 1, lw=1.5, color='k')
        if set_xticks:
            ax.set_xticks(range(0, len(ave_grads), 1))
            ax.set_xticklabels(layers, rotation='vertical', fontsize=7)
            ax.set_xlabel('Layers')
        ax.set_xlim(left=0, right=len(ave_grads))
        ax.set_ylim(bottom=-0.001, top=0.02) # Zoom in on the lower gradient regions
        if set_ylabels:
            ax.set_ylabel('Average gradient')
        ax.grid(True)

    @staticmethod
    def plot_gradient_flow_over_epochs(gradient_stats_entries, output_file_name):
        epoch_number, iteration_number = set(), set()
        for epoch, iteration, _ in gradient_stats_entries:
            epoch_number.add(epoch)
            iteration_number.add(iteration)

        epoch_number = len(epoch_number)
        iteration_number = len(iteration_number)

        fig, axs = plt.subplots(
            epoch_number,
            iteration_number,
            figsize=(epoch_number*8, iteration_number*8),
            sharey=True,
            sharex=True
        )
        k = 0
        for i in trange(epoch_number):
            for j in trange(iteration_number):
                _, _, gradient_stats_entry = gradient_stats_entries[k]
                GradientStats.plot_gradient_flow(
                    gradient_stats_entry['model'],
                    axs[i][j],
                    set_xticks=True if i + 1 == epoch_number else False,
                    set_ylabels=True if j == 0 else False
                )
                k += 1

        ConsoleLogger.status('Saving gradient flow plot...')
        fig.suptitle('Gradient flow', fontsize='x-large')
        fig.legend([
            Line2D([0], [0], color='c', lw=4),
            Line2D([0], [0], color='b', lw=4),
            Line2D([0], [0], color='k', lw=4)],
            ['max-gradient', 'mean-gradient', 'zero-gradient'],
            loc="center right", # Position of legend
            borderaxespad=0.1 # Small spacing around legend box
        )
        fig.savefig(output_file_name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close(fig)
