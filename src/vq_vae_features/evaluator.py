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

from dataset.spectrogram_parser import SpectrogramParser
from dataset.vctk import VCTK
from error_handling.console_logger import ConsoleLogger
from evaluation.alignment_stats import AlignmentStats

import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import numpy as np
import umap
from textwrap import wrap
import seaborn as sns
import textgrid
from tqdm import tqdm
import pickle


class Evaluator(object):

    def __init__(self, device, model, data_stream, configuration):
        self._device = device
        self._model = model
        self._data_stream = data_stream
        self._configuration = configuration
        self._vctk = VCTK(self._configuration['data_root'], ratio=self._configuration['train_val_split'])

    def evaluate(self, results_path, experiment_name, evaluation_options):
        self._model.eval()

        if evaluation_options['plot_comparaison_plot'] or \
            evaluation_options['plot_quantized_embedding_spaces'] or \
            evaluation_options['plot_distances_histogram']:
            evaluation_entry = self._evaluate_once(results_path, experiment_name)

        if evaluation_options['plot_comparaison_plot']:
            self._compute_comparaison_plot(evaluation_entry, results_path, experiment_name)

        if evaluation_options['plot_quantized_embedding_spaces']:
            self._plot_quantized_embedding_spaces(evaluation_entry, results_path, experiment_name)

        if evaluation_options['plot_distances_histogram']:
            self._plot_distances_histogram(evaluation_entry, results_path, experiment_name)

        #self._test_denormalization(evaluation_entry, results_path, experiment_name)

        if evaluation_options['compute_many_to_one_mapping']:
            self._many_to_one_mapping(results_path, experiment_name)

        if evaluation_options['compute_speaker_dependency_stats']:
            self._compute_speaker_dependency_stats(results_path, experiment_name)

        if evaluation_options['compute_alignments'] or evaluation_options['compute_clustering_metrics']:
            alignment_stats = AlignmentStats(
                self._data_stream,
                self._vctk,
                self._configuration,
                self._device,
                self._model,
                results_path,
                experiment_name
            )
        if evaluation_options['compute_alignments']:
            if not os.path.isfile(results_path + os.sep + 'vctk_groundtruth_alignments.pickle'):
                alignment_stats.compute_groundtruth_alignments()
                alignment_stats.compute_groundtruth_bigrams_matrix(wo_diag=True)
                alignment_stats.compute_groundtruth_bigrams_matrix(wo_diag=False)
                alignment_stats.compute_groundtruth_phonemes_frequency()
            else:
                ConsoleLogger.status('Groundtruth alignments already exist')

            if not os.path.isfile(results_path + os.sep + experiment_name + '_vctk_empirical_alignments.pickle'):
                alignment_stats.compute_empirical_alignments()
                alignment_stats.compute_empirical_bigrams_matrix(wo_diag=True)
                alignment_stats.compute_empirical_bigrams_matrix(wo_diag=False)
                alignment_stats.comupte_empirical_encodings_frequency()
            else:
                ConsoleLogger.status('Empirical alignments already exist')

        if evaluation_options['compute_clustering_metrics']:
            alignment_stats.compute_clustering_metrics()

    def _evaluate_once(self, results_path, experiment_name):
        self._model.eval()

        data = next(iter(self._data_stream.validation_loader))

        preprocessed_audio = data['preprocessed_audio'].to(self._device)
        valid_originals = data['input_features'].to(self._device)
        speaker_ids = data['speaker_id'].to(self._device)
        target = data['output_features'].to(self._device)
        wav_filename = data['wav_filename']
        shifting_time = data['shifting_time'].to(self._device)
        preprocessed_length = data['preprocessed_length'].to(self._device)

        valid_originals = valid_originals.permute(0, 2, 1).contiguous().float()
        batch_size = valid_originals.size(0)
        target = _target.permute(0, 2, 1).contiguous().float()
        wav_filename = wav_filename[0][0]

        z = self._model.encoder(valid_originals)
        z = self._model.pre_vq_conv(z)
        _, quantized, _, encodings, distances, encoding_indices, _, \
            encoding_distances, embedding_distances, frames_vs_embedding_distances, \
            concatenated_quantized = self._model.vq(z)
        valid_reconstructions = self._model.decoder(quantized, data_stream.speaker_dic, speaker_ids)[0]

        return {
            'preprocessed_audio': preprocessed_audio,
            'valid_originals': valid_originals,
            'speaker_ids': speaker_ids,
            'target': target,
            'wav_filename': wav_filename,
            'shifting_time': shifting_time,
            'preprocessed_length': preprocessed_length,
            'batch_size': batch_size,
            'quantized': quantized,
            'encodings': encodings,
            'distances': distances,
            'encoding_indices': encoding_indices,
            'encoding_distances': encoding_distances,
            'embedding_distances': embedding_distances,
            'frames_vs_embedding_distances': frames_vs_embedding_distances,
            'concatenated_quantized': concatenated_quantized,
            'valid_reconstructions': valid_reconstructions
        }

    def _compute_comparaison_plot(self, evaluation_entry, results_path, experiment_name):
        utterence_key = evaluation_entry['wav_filename'].split('/')[-1].replace('.wav', '')
        utterence = self._vctk.utterences[utterence_key].replace('\n', '')
        phonemes_alignment_path = os.sep.join(evaluation_entry['wav_filename'].split('/')[:-3]) \
            + os.sep + 'phonemes' + os.sep + utterence_key.split('_')[0] + os.sep \
            + utterence_key + '.TextGrid'
        #tg = textgrid.TextGrid()
        #tg.read(phonemes_alignment_path)
        #for interval in tg.tiers[0]:
    
        ConsoleLogger.status('Original utterence: {}'.format(utterence))

        if self._configuration['verbose']:
            ConsoleLogger.status('utterence: {}'.format(utterence))

        spectrogram_parser = SpectrogramParser()
        preprocessed_audio = evaluation_entry['preprocessed_audio'].detach().cpu()[0].numpy().squeeze()
        spectrogram = spectrogram_parser.parse_audio(preprocessed_audio).contiguous()

        spectrogram = spectrogram.detach().cpu().numpy()

        valid_originals = evaluation_entry['valid_originals'].detach().cpu()[0].numpy()

        probs = F.softmax(-evaluation_entry['distances'][0], dim=1).detach().cpu().transpose(0, 1).contiguous()

        #target = self._target.detach().cpu()[0].numpy()

        valid_reconstructions = evaluation_entry['valid_reconstructions'].detach().cpu().numpy()

        fig, axs = plt.subplots(6, 1, figsize=(35, 30), sharex=True)

        # Waveform of the original speech signal
        axs[0].set_title('Waveform of the original speech signal')
        axs[0].plot(np.arange(len(preprocessed_audio)) / float(self._configuration['sampling_rate']), preprocessed_audio)

        # TODO: Add number of encoding indices at the same rate of the tokens with _compute_unified_time_scale()
        """
        # Example of vertical red lines
        xposition = [0.3, 0.4, 0.45]
        for xc in xposition:
            plt.axvline(x=xc, color='r', linestyle='-', linewidth=1)
        """

        # Spectrogram of the original speech signal
        axs[1].set_title('Spectrogram of the original speech signal')
        self._plot_pcolormesh(spectrogram, fig, x=self._compute_unified_time_scale(spectrogram.shape[1]), axis=axs[1])

        # MFCC + d + a of the original speech signal
        axs[2].set_title('Augmented MFCC + d + a #filters=13+13+13 of the original speech signal')
        self._plot_pcolormesh(valid_originals, fig, x=self._compute_unified_time_scale(valid_originals.shape[1]), axis=axs[2])

        # Softmax of distances computed in VQ
        axs[3].set_title('Softmax of distances computed in VQ\n($||z_e(x) - e_i||^2_2$ with $z_e(x)$ the output of the encoder prior to quantization)')
        self._plot_pcolormesh(probs, fig, x=self._compute_unified_time_scale(probs.shape[1], downsampling_factor=2), axis=axs[3])

        encodings = self._encodings.detach().cpu().numpy()
        axs[4].set_title('Encodings')
        self._plot_pcolormesh(encodings[0].transpose(), fig, x=self._compute_unified_time_scale(encodings[0].transpose().shape[1],
            downsampling_factor=2), axis=axs[4])

        # Actual reconstruction
        axs[5].set_title('Actual reconstruction')
        self._plot_pcolormesh(valid_reconstructions, fig, x=self._compute_unified_time_scale(valid_reconstructions.shape[1]), axis=axs[5])

        output_path = results_path + os.sep + experiment_name + '_evaluation-comparaison-plot.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _plot_pcolormesh(self, data, fig, x=None, y=None, axis=None):
        axis = plt.gca() if axis is None else axis # default axis if None
        x = np.arange(data.shape[1]) if x is None else x # default x shape if None
        y = np.arange(data.shape[0]) if y is None else y # default y shape if None
        c = axis.pcolormesh(x, y, data)
        fig.colorbar(c, ax=axis)

    def _plot_quantized_embedding_spaces(self, evaluation_entry, results_path, experiment_name):
        # TODO: do it for more that one sample

        concatenated_quantized = evaluation_entry['concatenated_quantized'].detach().cpu().numpy()
        embedding = self._model.vq.embedding.weight.data.cpu().detach().numpy()
        n_embedding = embedding.shape[0]
        encoding_indices = evaluation_entry['encoding_indices'].detach().cpu().numpy()
        encoding_indices = np.concatenate(encoding_indices)
        quantized_embedding_space = np.concatenate(
            (concatenated_quantized, embedding)
        )
        speaker_ids = evaluation_entry['speaker_ids'].detach().cpu().numpy()
        time_speaker_ids = np.repeat(
            speaker_ids,
            concatenated_quantized.shape[0] // self._data_stream.validation_batch_size,
            axis=1
        )
        time_speaker_ids = np.concatenate(time_speaker_ids)

        for n_neighbors in [3, 10]:
            mapping = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=0.0,
                metric='euclidean'
            )

            projection = mapping.fit_transform(quantized_embedding_space)

            self._plot_quantized_embedding_space(projection, n_neighbors, n_embedding,
                time_speaker_ids, encoding_indices, results_path, experiment_name, cmap='cubehelix')

    def _compute_unified_time_scale(self, shape, winstep=0.01, downsampling_factor=1):
        return np.arange(shape) * winstep * downsampling_factor

    def _plot_quantized_embedding_space(self, projection, n_neighbors, n_embedding,
        time_speaker_ids, encoding_indices, results_path, experiment_name, cmap='tab20',
        xlabel_width=60):

        def _configure_ax(ax, title=None, xlabel=None, ylabel=None, legend=False):
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

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

        # Colored by speaker id
        #axs[0].scatter(projection[:-n_embedding,0], projection[:-n_embedding, 1], s=10, c=time_speaker_ids, cmap=cmap) # audio frame colored by speaker id
        axs[0] = self._jittered_scatter(axs[0], projection[:-n_embedding,0], projection[:-n_embedding, 1], s=10, c=time_speaker_ids, cmap=cmap)
        axs[0].scatter(projection[-n_embedding:,0], projection[-n_embedding:, 1], s=50, marker='x', c='k', alpha=0.8) # embedding
        xlabel = 'Embedding of ' + str(self._data_stream.validation_batch_size) + ' valuations' \
            ' with ' + str(n_neighbors) + ' neighbors with the audio frame points colored' \
            ' by speaker id and the embedding marks colored in black'
        axs[0] = _configure_ax(
            axs[0],
            xlabel='\n'.join(wrap(xlabel, xlabel_width))
        )

        # Colored by encoding indices
        #axs[1].scatter(projection[:-n_embedding,0], projection[:-n_embedding, 1], s=10, c=encoding_indices, cmap=cmap) # audio frame colored by encoding indices
        axs[1] = self._jittered_scatter(axs[1], projection[:-n_embedding,0], projection[:-n_embedding, 1], s=10, c=encoding_indices, cmap=cmap)
        axs[1].scatter(projection[-n_embedding:,0], projection[-n_embedding:, 1], s=50, marker='x', c=np.arange(n_embedding), cmap=cmap) # different color for each embedding
        xlabel = 'Embedding of ' + str(self._data_stream.validation_batch_size) + ' valuations' \
            ' with ' + str(n_neighbors) + ' neighbors with the audio frame points colored' \
            ' by encoding indices and the embedding marks colored by number of embedding' \
            ' vectors (using the same color map)'
        axs[1] = _configure_ax(
            axs[1],
            xlabel='\n'.join(wrap(xlabel, width=xlabel_width))
        )

        plt.suptitle('Quantized embedding space of ' + experiment_name, fontsize=16)

        output_path = results_path + os.sep + experiment_name + '_quantized_embedding_space-n' + str(n_neighbors) + '.png'
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _plot_distances_histogram(self, evaluation_entry, results_path, experiment_name):
        encodings_distances = evaluation_entry['encoding_distances'][0].detach().cpu().numpy()
        embeddings_distances = evaluation_entry['embedding_distances'].detach().cpu().numpy()
        frames_vs_embedding_distances = evaluation_entry['frames_vs_embedding_distances'].detach()[0].cpu().transpose(0, 1).numpy().ravel()

        if self._configuration['verbose']:
            ConsoleLogger.status('encoding_distances[0].size(): {}'.format(encoding_distances.shape))
            ConsoleLogger.status('embedding_distances.size(): {}'.format(embedding_distances.shape))
            ConsoleLogger.status('frames_vs_embedding_distances[0].shape: {}'.format(frames_vs_embedding_distances.shape))

        fig, axs = plt.subplots(3, 1, figsize=(30, 20), sharex=True)

        axs[0].set_title('\n'.join(wrap('Histogram of the distances between the'
            ' encodings vectors', 60)))
        sns.distplot(encodings_distances, hist=True, kde=False, ax=axs[0], norm_hist=True)

        axs[1].set_title('\n'.join(wrap('Histogram of the distances between the'
            ' embeddings vectors', 60)))
        sns.distplot(embeddings_distances, hist=True, kde=False, ax=axs[1], norm_hist=True)

        axs[2].set_title(
            'Histogram of the distances computed in'
            ' VQ\n($||z_e(x) - e_i||^2_2$ with $z_e(x)$ the output of the encoder'
            ' prior to quantization)'
        )
        sns.distplot(frames_vs_embedding_distances, hist=True, kde=False, ax=axs[2], norm_hist=True)

        output_path = results_path + os.sep + experiment_name + '_distances-histogram-plot.png'
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _test_denormalization(self, evaluation_entry, results_path, experiment_name):
        valid_originals = evaluation_entry['valid_originals'].detach().cpu()[0].numpy()
        valid_reconstructions = evaluation_entry['valid_reconstructions'].detach().cpu().numpy()
        normalizer = self._data_stream.normalizer

        denormalized_valid_originals = (normalizer['train_std'] * valid_originals.transpose() + normalizer['train_mean']).transpose()
        denormalized_valid_reconstructions = (normalizer['train_std'] * valid_reconstructions.transpose() + normalizer['train_mean']).transpose()

        # TODO: Remove the deltas and the accelerations, remove the zeros because it's the
        # energy, and compute the distance between the two

        fig, axs = plt.subplots(4, 1, figsize=(30, 20), sharex=True)

        # MFCC + d + a of the original speech signal
        axs[0].set_title('Augmented MFCC + d + a #filters=13+13+13 of the original speech signal')
        self._plot_pcolormesh(valid_originals, fig, x=self._compute_unified_time_scale(valid_originals.shape[1]), axis=axs[0])

        # Actual reconstruction
        axs[1].set_title('Actual reconstruction')
        self._plot_pcolormesh(valid_reconstructions, fig, x=self._compute_unified_time_scale(valid_reconstructions.shape[1]), axis=axs[1])

        # Denormalization of the original speech signal
        axs[2].set_title('Denormalized target')
        self._plot_pcolormesh(denormalized_valid_originals, fig, x=self._compute_unified_time_scale(denormalized_valid_originals.shape[1]), axis=axs[2])

        # Denormalization of the original speech signal
        axs[3].set_title('Denormalized reconstruction')
        self._plot_pcolormesh(denormalized_valid_reconstructions, fig, x=self._compute_unified_time_scale(denormalized_valid_reconstructions.shape[1]), axis=axs[3])

        output_path = results_path + os.sep + experiment_name + '_test-denormalization-plot.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _jittered_scatter(self, ax, x, y, cmap, c, s, alpha=None, marker=None):

        def rand_jitter(arr):
            stdev = .01*(max(arr) - min(arr))
            return arr + np.random.randn(len(arr)) * stdev

        ax.scatter(rand_jitter(x), rand_jitter(y), cmap=cmap, c=c, alpha=alpha, s=s, marker=marker)
        return ax

    def _many_to_one_mapping(self, results_path, experiment_name):
        # TODO: fix it for batch size greater than one

        tokens_selections = list()
        val_speaker_ids = set()

        with tqdm(self._data_stream.validation_loader) as bar:
            for data in bar:
                valid_originals = data['input_features'].to(self._device).permute(0, 2, 1).contiguous().float()
                speaker_ids = data['speaker_id'].to(self._device)
                shifting_times = data['shifting_time'].to(self._device)
                wav_filenames = data['wav_filename']

                speaker_id = wav_filenames[0][0].split(os.sep)[-2]
                val_speaker_ids.add(speaker_id)

                if speaker_id not in os.listdir(self._vctk.raw_folder + os.sep + 'VCTK-Corpus' + os.sep + 'phonemes'):
                    # TODO: log the missing folders
                    continue

                z = self._model.encoder(valid_originals)
                z = self._model.pre_vq_conv(z)
                _, quantized, _, encodings, _, encoding_indices, _, \
                    _, _, _, _ = self._model.vq(z)
                valid_reconstructions = self._model.decoder(quantized, self._data_stream.speaker_dic, speaker_ids)
                B = valid_reconstructions.size(0)

                encoding_indices = encoding_indices.view(B, -1, 1)

                for i in range(len(valid_reconstructions)):
                    wav_filename = wav_filenames[0][i]
                    utterence_key = wav_filename.split('/')[-1].replace('.wav', '')
                    phonemes_alignment_path = os.sep.join(wav_filename.split('/')[:-3]) + os.sep + 'phonemes' + os.sep + utterence_key.split('_')[0] + os.sep \
                        + utterence_key + '.TextGrid'
                    tg = textgrid.TextGrid()
                    tg.read(phonemes_alignment_path)
                    entry = {
                        'encoding_indices': encoding_indices[i].detach().cpu().numpy(),
                        'groundtruth': tg.tiers[1],
                        'shifting_time': shifting_times[i].detach().cpu().item()
                    }
                    tokens_selections.append(entry)

        ConsoleLogger.status(val_speaker_ids)

        ConsoleLogger.status('{} tokens selections retreived'.format(len(tokens_selections)))

        phonemes_mapping = dict()
        # For each tokens selections (i.e. the number of valuations)
        for entry in tokens_selections:
            encoding_indices = entry['encoding_indices']
            unified_encoding_indices_time_scale = self._compute_unified_time_scale(
                encoding_indices.shape[0], downsampling_factor=2) # Compute the time scale array for each token
            """
            Search the grountruth phoneme where the selected token index time scale
            is within the groundtruth interval.
            Then, it adds the selected token index in the list of indices selected for
            the a specific token in the tokens mapping dictionnary.
            """
            for i in range(len(unified_encoding_indices_time_scale)):
                index_time_scale = unified_encoding_indices_time_scale[i] + entry['shifting_time']
                corresponding_phoneme = None
                for interval in entry['groundtruth']:
                    # TODO: replace that by nearest interpolation
                    if index_time_scale >= interval.minTime and index_time_scale <= interval.maxTime:
                        corresponding_phoneme = interval.mark
                        break
                if not corresponding_phoneme:
                    ConsoleLogger.warn("Corresponding phoneme not found. unified_encoding_indices_time_scale[{}]: {}"
                        "entry['shifting_time']: {} index_time_scale: {}".format(i, unified_encoding_indices_time_scale[i],
                        entry['shifting_time'], index_time_scale))
                if corresponding_phoneme not in phonemes_mapping:
                    phonemes_mapping[corresponding_phoneme] = list()
                phonemes_mapping[corresponding_phoneme].append(encoding_indices[i][0])

        ConsoleLogger.status('phonemes_mapping: {}'.format(phonemes_mapping))

        tokens_mapping = dict() # dictionnary that will contain the distribution for each token to fits with a certain phoneme

        """
        Fill the tokens_mapping such that for each token index (key)
        it contains the list of tuple of (phoneme, prob) where prob
        is the probability that the token fits this phoneme.
        """
        for phoneme, indices in phonemes_mapping.items():
            for index in list(set(indices)):
                if index not in tokens_mapping:
                    tokens_mapping[index] = list()
                tokens_mapping[index].append((phoneme, indices.count(index) / len(indices)))

        # Sort the probabilities for each token 
        for index, distribution in tokens_mapping.items():
            tokens_mapping[index] = list(sorted(distribution, key = lambda x: x[1], reverse=True))

        ConsoleLogger.status('tokens_mapping: {}'.format(tokens_mapping))

        with open(results_path + os.sep + experiment_name + '_phonemes_mapping.pickle', 'wb') as f:
            pickle.dump(phonemes_mapping, f)

        with open(results_path + os.sep + experiment_name + '_tokens_mapping.pickle', 'wb') as f:
            pickle.dump(tokens_mapping, f)

    def _compute_speaker_dependency_stats(self, results_path, experiment_name):
        """
        The goal of this function is to investiguate wether or not the supposed
        phonemes stored in the embeddings space are speaker independents.
        The algorithm is as follow:
            - Evaluate the model using the val dataset. Save each resulting
              embedding, with the corresponding speaker;
            - Group the embeddings by speaker;
            - Compute the distribution of each embedding;
            - Compute all the distances between all possible distribution couples, using
              a distribution distance (e.g. entropy) and plot them.
        """
        all_speaker_ids = list()
        all_embeddings = torch.tensor([]).to(self._device)

        with tqdm(self._data_stream.validation_loader) as bar:
            for data in bar:
                valid_originals = data['input_features'].to(self._device).permute(0, 2, 1).contiguous().float()
                speaker_ids = data['speaker_id'].to(self._device)
                wav_filenames = data['wav_filename']

                z = self._model.encoder(valid_originals)
                z = self._model.pre_vq_conv(z)
                _, quantized, _, _, _, _, _, \
                    _, _, _, _ = self._model.vq(z)
                valid_reconstructions = self._model.decoder(quantized, self._data_stream.speaker_dic, speaker_ids)
                B = valid_reconstructions.size(0)

                all_speaker_ids.append(speaker_ids.detach().cpu().numpy().tolist())
                #torch.cat(all_embeddings, self._model.vq.embedding.weight.data) # FIXME

        # - Group the embeddings by speaker: create a tensor/numpy per speaker id from all_embeddings
        # - Compute the distribution of each embedding (seaborn histogram, softmax)
        # - Compute all the distances between all possible distribution couples, using
        #   a distribution distance (e.g. entropy) and plot them (seaborn histogram?)

        # Snippet
        #_embedding_distances = [torch.dist(items[0], items[1], 2).to(self._device) for items in combinations(self._embedding.weight, r=2)]
        #embedding_distances = torch.tensor(_embedding_distances).to(self._device)    
