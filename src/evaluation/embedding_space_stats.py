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
from evaluation.utils import Utils

import numpy as np
import umap
import matplotlib.pyplot as plt
from matplotlib.image import imread
from textwrap import wrap
import os
from tqdm import tqdm
import pickle
import warnings


class EmbeddingSpaceStats(object):

    def __init__(self, results_path, experiment_name, cmap='cubehelix', all_n_neighbors=[3, 10]):
        self._results_path = results_path
        self._experiment_name = experiment_name
        self._cmap = cmap
        self._all_n_neighbors = all_n_neighbors

    def compute_quantized_embedding_space_projections(self, quantized_embedding_space_state):
        projections = list()

        for n_neighbors in self._all_n_neighbors:
            mapping = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=0.0,
                metric='euclidean'
            )

            projection = mapping.fit_transform(quantized_embedding_space_state['quantized_embedding_space'])
            projections.append((projection, n_neighbors))

        return projections

    def plot_quantized_embedding_spaces(self, projections, quantized_embedding_space_state):
        for projection in projections:
            self.plot_quantized_embedding_space(
                projection[0],
                projection[1],
                quantized_embedding_space_state['n_embedding'],
                quantized_embedding_space_state['time_speaker_ids'],
                quantized_embedding_space_state['encoding_indices'],
                quantized_embedding_space_state['batch_size']
            )

    def plot_quantized_embedding_space(self, projection, n_neighbors, n_embedding,
        time_speaker_ids, encoding_indices, batch_size, use_jittered_scatter=True,
        xlabel_width=60):

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

        # Colored by speaker id
        if use_jittered_scatter:
            axs[0] = self._jittered_scatter(axs[0], projection[:-n_embedding,0], projection[:-n_embedding, 1], s=10, c=time_speaker_ids, cmap=self._cmap)
        else:
            axs[0].scatter(projection[:-n_embedding,0], projection[:-n_embedding, 1], s=10, c=time_speaker_ids, cmap=self._cmap) # audio frame colored by speaker id
        axs[0].scatter(projection[-n_embedding:,0], projection[-n_embedding:, 1], s=50, marker='x', c='k', alpha=0.8) # embedding
        xlabel = 'Embedding of ' + str(batch_size) + ' valuations' \
            ' with ' + str(n_neighbors) + ' neighbors with the audio frame points colored' \
            ' by speaker id and the embedding marks colored in black'
        axs[0] = self._configure_ax(
            axs[0],
            xlabel='\n'.join(wrap(xlabel, xlabel_width))
        )

        # Colored by encoding indices
        if use_jittered_scatter:
            axs[1] = self._jittered_scatter(axs[1], projection[:-n_embedding,0], projection[:-n_embedding, 1], s=10, c=encoding_indices, cmap=self._cmap)
        else:
            axs[1].scatter(projection[:-n_embedding,0], projection[:-n_embedding, 1], s=10, c=encoding_indices, cmap=cmap) # audio frame colored by encoding indices
        axs[1].scatter(projection[-n_embedding:, 0], projection[-n_embedding:, 1], s=50, marker='x', c=np.arange(n_embedding), cmap=self._cmap) # different color for each embedding
        xlabel = 'Embedding of ' + str(batch_size) + ' valuations' \
            ' with ' + str(n_neighbors) + ' neighbors with the audio frame points colored' \
            ' by encoding indices and the embedding marks colored by number of embedding' \
            ' vectors (using the same color map)'
        axs[1] = self._configure_ax(
            axs[1],
            xlabel='\n'.join(wrap(xlabel, width=xlabel_width))
        )

        plt.suptitle('Quantized embedding space of ' + self._experiment_name, fontsize=16)

        output_path = self._results_path + os.sep + self._experiment_name + '_quantized_embedding_space-n' + str(n_neighbors) + '.png'
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    @staticmethod
    def compute_quantized_embedding_space_state(evaluation_entry, embedding, batch_size):
        concatenated_quantized = evaluation_entry['concatenated_quantized'].detach().cpu().numpy()
        embedding = embedding.weight.data.cpu().detach().numpy()
        n_embedding = embedding.shape[0]
        encoding_indices = evaluation_entry['encoding_indices'].detach().cpu().numpy()
        encoding_indices = np.concatenate(encoding_indices)
        quantized_embedding_space = np.concatenate(
            (concatenated_quantized, embedding)
        )
        speaker_ids = evaluation_entry['speaker_ids'].detach().cpu().numpy()
        time_speaker_ids = np.repeat(
            speaker_ids,
            concatenated_quantized.shape[0] // batch_size,
            axis=1
        )
        time_speaker_ids = np.concatenate(time_speaker_ids)

        quantized_embedding_space_state = {
            'quantized_embedding_space': quantized_embedding_space,
            'n_embedding': n_embedding,
            'encoding_indices': encoding_indices,
            'time_speaker_ids': time_speaker_ids,
            'batch_size': batch_size
        }

        return quantized_embedding_space_state

    @staticmethod
    def compute_and_plot_quantized_embedding_space_projections(results_path, experiment_name,
        evaluation_entry, embedding, batch_size):
        embedding_space_stats = EmbeddingSpaceStats(results_path, experiment_name)
        quantized_embedding_space_state = EmbeddingSpaceStats.compute_quantized_embedding_space_state(
            evaluation_entry,
            embedding,
            batch_size
        )
        projections = embedding_space_stats.compute_quantized_embedding_space_projections(quantized_embedding_space_state)
        embedding_space_stats.plot_quantized_embedding_spaces(projections, quantized_embedding_space_state)

    @staticmethod
    def compute_quantized_embedding_spaces_animation(all_experiments_paths, all_experiments_names,
        all_results_paths):

        for i in range(len(all_experiments_paths)):
            experiment_path = all_experiments_paths[i]
            experiment_name = all_experiments_names[i]
            experiment_results_path = all_results_paths[i]
            # List all file names related to the codebook stats for the current observed experiment
            file_names = [file_name for file_name in os.listdir(experiment_path) if 'codebook-stats' in file_name and experiment_name in file_name]

            # Sort file names by epoch number and iteration number as well
            file_names = sorted(file_names, key=lambda x: 
                (int(x.replace(experiment_name + '_', '').replace('_codebook-stats.pickle', '').split('_')[0]),
                int(x.replace(experiment_name + '_', '').replace('_codebook-stats.pickle', '').split('_')[1]))
            )

            projections = list()

            with tqdm(file_names) as bar:
                bar.set_description('Processing projections')
                for file_name in bar:
                    codebook_stats_entry = None
                    with open(experiment_path + os.sep + file_name, 'rb') as file:
                        codebook_stats_entry = pickle.load(file)
                    concatenated_quantized = codebook_stats_entry['concatenated_quantized']
                    embedding = codebook_stats_entry['embedding']
                    n_embedding = codebook_stats_entry['n_embedding']
                    encoding_indices = np.concatenate(codebook_stats_entry['encoding_indices'])
                    quantized_embedding_space = np.concatenate(
                        (concatenated_quantized, embedding)
                    )
                    speaker_ids = codebook_stats_entry['speaker_ids']
                    time_speaker_ids = np.repeat(
                        speaker_ids,
                        concatenated_quantized.shape[0] // 2,
                        #concatenated_quantized.shape[0] // codebook_stats_entry['batch_size'],
                        axis=1
                    )
                    time_speaker_ids = np.concatenate(time_speaker_ids)
                    n_neighbors = 3
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        mapping = umap.UMAP(
                            n_neighbors=n_neighbors,
                            min_dist=0.0,
                            metric='euclidean'
                        )
                    projection = mapping.fit_transform(quantized_embedding_space)
                    projections.append({
                        'projection': projection,
                        'n_neighbors': n_neighbors,
                        'n_embedding': n_embedding,
                        'time_speaker_ids': time_speaker_ids,
                        'encoding_indices': encoding_indices
                    })
                    bar.update(1)    

            cmap = 'cubehelix'
            projected_images = list()

            with tqdm(projections) as bar:
                bar.set_description('Plotting projections')
                for projection_entry in bar:
                    projection = projection_entry['projection']
                    n_embedding = projection_entry['n_embedding']
                    time_speaker_ids = projection_entry['time_speaker_ids']
                    encoding_indices = projection_entry['encoding_indices']

                    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

                    axs[0].scatter(projection[:-n_embedding, 0], projection[:-n_embedding, 1], s=10, c=time_speaker_ids, cmap=cmap) # audio frame colored by speaker id
                    axs[0].scatter(projection[-n_embedding:, 0], projection[-n_embedding:, 1], s=50, marker='x', c='k', alpha=0.8) # embedding

                    axs[1].scatter(projection[:-n_embedding,0], projection[:-n_embedding, 1], s=10, c=encoding_indices, cmap=cmap) # audio frame colored by encoding indices
                    axs[1].scatter(projection[-n_embedding:, 0], projection[-n_embedding:, 1], s=50, marker='x', c=np.arange(n_embedding), cmap=cmap) # different color for each embedding

                    projection_file_path = '..' + os.sep + '_tmp_projection' + '.png'
                    fig.savefig(projection_file_path)
                    plt.close(fig)
                    projected_images.append(imread(projection_file_path))
                    os.remove(projection_file_path)
                    bar.update(1)

            ConsoleLogger.status('Building gif from projected images...')
            Utils.build_gif(projected_images)

    def _configure_ax(self, ax, title=None, xlabel=None, ylabel=None, legend=False):
        ax.minorticks_off()
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend:
            ax.legend()
        ax.grid(True)
        ax.margins(x=0)
        return ax

    def _jittered_scatter(self, ax, x, y, cmap, c, s, alpha=None, marker=None):

        def rand_jitter(arr):
            stdev = .01*(max(arr) - min(arr))
            return arr + np.random.randn(len(arr)) * stdev

        ax.scatter(rand_jitter(x), rand_jitter(y), cmap=cmap, c=c, alpha=alpha, s=s, marker=marker)
        return ax
