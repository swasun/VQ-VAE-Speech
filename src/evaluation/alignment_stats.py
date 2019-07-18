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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import numpy as np
import textgrid
from tqdm import tqdm
import pickle
from sklearn.preprocessing import normalize
import sklearn
import json
from itertools import cycle


class AlignmentStats(object):
    
    def __init__(self, data_stream, vctk, configuration, device, model,
        results_path, experiment_name, alignment_subset):

        self._data_stream = data_stream
        self._vctk = vctk
        self._configuration = configuration
        self._device = device
        self._model = model
        self._results_path = results_path
        self._experiment_name = experiment_name
        self._alignment_subset = alignment_subset

        self._model.eval()

    def compute_groundtruth_alignments(self):
        ConsoleLogger.status('Computing groundtruth alignments of VCTK val dataset...')

        desired_time_interval = 0.02
        extended_alignment_dataset = list()
        possible_phonemes = set()
        phonemes_counter = dict()
        total_phonemes_apparations = 0
        data_length = self._configuration['length'] / self._configuration['sampling_rate']
        speaker_id_folders = os.listdir(self._vctk.raw_folder + os.sep + 'VCTK-Corpus' + os.sep + 'phonemes')
        loader = self._data_stream.training_loader if self._alignment_subset == 'train' else self._data_stream.validation_loader

        with tqdm(loader) as bar:
            for data in bar:
                speaker_ids = data['speaker_id'].to(self._device)
                wav_filenames = data['wav_filename']
                shifting_times = data['shifting_time'].to(self._device)
                loader_indices = data['index'].to(self._device)

                speaker_id = wav_filenames[0][0].split('/')[-2]
                if speaker_id not in speaker_id_folders:
                    # TODO: log the missing folders
                    #ConsoleLogger.warn('speaker id {} not found'.format(speaker_id))
                    continue

                for i in range(len(shifting_times)):
                    wav_filename = wav_filenames[0][i]
                    utterence_key = wav_filename.split('/')[-1].replace('.wav', '')
                    phonemes_alignment_path = os.sep.join(wav_filename.split('/')[:-3]) + os.sep + 'phonemes' + os.sep + utterence_key.split('_')[0] + os.sep \
                        + utterence_key + '.TextGrid'
                    if not os.path.isfile(phonemes_alignment_path):
                        # TODO: log this warn instead of print it
                        #ConsoleLogger.warn('File {} not found'.format(phonemes_alignment_path))
                        break

                    shifting_time = shifting_times[i].detach().cpu().item()
                    target_time_scale = np.arange((data_length / desired_time_interval) + 1) * desired_time_interval + shifting_time
                    shifted_indices = np.where(target_time_scale >= shifting_time)
                    tg = textgrid.TextGrid()
                    tg.read(phonemes_alignment_path)
                    """if target_time_scale[-1] > tg.tiers[1][-1].maxTime:
                        ConsoleLogger.error('Shifting time error at {}.pickle: shifting_time:{}' \
                            ' target_time_scale[-1]:{} > tg.tiers[1][-1].maxTime:{}\n'
                            'wav filename:{} phonemes alignment path:{}'.format(
                            loader_indices[i].detach().cpu().item(),
                            shifting_time,
                            target_time_scale[-1],
                            tg.tiers[1][-1].maxTime,
                            wav_filename, phonemes_alignment_path))
                        continue"""

                    phonemes = list()
                    current_target_time_index = 0
                    for interval in tg.tiers[1]:
                        if interval.mark in ['', '-', "'"]:
                            if interval == tg.tiers[1][-1] and len(phonemes) != int(data_length / desired_time_interval):
                                previous_interval = tg.tiers[1][-2]
                                ConsoleLogger.warn("{}/{} phonemes aligned. Add the last valid phoneme '{}' in the list to have the correct number.\n"
                                    "Sanity checks to find the possible cause:\n"
                                    "current_target_time_index < (data_length / desired_time_interval): {}\n"
                                    "target_time_scale[current_target_time_index] >= interval.minTime: {}\n"
                                    "target_time_scale[current_target_time_index] <= interval.maxTime: {}".format(
                                    len(phonemes), int(data_length / desired_time_interval), previous_interval.mark,
                                    current_target_time_index < (data_length / desired_time_interval),
                                    target_time_scale[current_target_time_index] >= previous_interval.minTime,
                                    target_time_scale[current_target_time_index] <= previous_interval.maxTime
                                ))
                                phonemes.append(previous_interval.mark)
                            continue
                        interval.minTime = float(interval.minTime)
                        interval.maxTime = float(interval.maxTime)
                        if interval.maxTime < shifting_time:
                            continue
                        interval.mark = interval.mark[:-1] if interval.mark[-1].isdigit() else interval.mark
                        possible_phonemes.add(interval.mark)
                        if interval.mark not in phonemes_counter:
                            phonemes_counter[interval.mark] = 0
                        phonemes_counter[interval.mark] += 1
                        total_phonemes_apparations += 1
                        while current_target_time_index < (data_length / desired_time_interval) and \
                            target_time_scale[current_target_time_index] >= interval.minTime and \
                            target_time_scale[current_target_time_index] <= interval.maxTime:
                            phonemes.append(interval.mark)
                            current_target_time_index += 1
                        if len(phonemes) == int(data_length / desired_time_interval):
                            break
                    if len(phonemes) != int(data_length / desired_time_interval):
                        intervals = ['min:{} max:{} mark:{}'.format(interval.minTime, interval.maxTime, interval.mark) for interval in tg.tiers[1]]
                        ConsoleLogger.error('Error - min:{} max:{} shifting:{} target_time_scale: {} intervals: {}\n'
                            '#phonemes:{} phonemes:{}\n'
                            'wav filename:{} phonemes alignment path:{}'.format(
                            interval.minTime, interval.maxTime, shifting_time, target_time_scale, intervals,
                            len(phonemes), phonemes, wav_filename, phonemes_alignment_path))
                    else:
                        extended_alignment_dataset.append((utterence_key, phonemes))

        groundtruth_alignments_path = self._results_path + os.sep + \
            'vctk_{}_groundtruth_alignments.pickle'.format(self._alignment_subset)
        with open(groundtruth_alignments_path, 'wb') as f:
            pickle.dump({
                'desired_time_interval': desired_time_interval,
                'extended_alignment_dataset': extended_alignment_dataset,
                'possible_phonemes': list(possible_phonemes),
                'phonemes_counter': phonemes_counter,
                'total_phonemes_apparations': total_phonemes_apparations
            }, f)

    def compute_groundtruth_bigrams_matrix(self, wo_diag=True):
        ConsoleLogger.status('Computing groundtruth bigrams matrix of VCTK val dataset {}...'.format(
            'without diagonal' if wo_diag else 'with diagonal'
        ))

        alignments_dic = None
        groundtruth_alignments_path = self._results_path + os.sep + \
            'vctk_{}_groundtruth_alignments.pickle'.format(self._alignment_subset)
        with open(groundtruth_alignments_path, 'rb') as f:
            alignments_dic = pickle.load(f)

        desired_time_interval = alignments_dic['desired_time_interval']
        extended_alignment_dataset = alignments_dic['extended_alignment_dataset']
        possible_phonemes = alignments_dic['possible_phonemes']
        phonemes_counter = alignments_dic['phonemes_counter']
        total_phonemes_apparations = alignments_dic['total_phonemes_apparations']

        possibles_phonemes_number = len(possible_phonemes)
        #ConsoleLogger.status('List of phonemes: {}'.format(possible_phonemes)) # TODO: log it instead of print it
        #ConsoleLogger.status('Number of phonemes: {}'.format(possibles_phonemes_number)) # TODO: log it instead of print it
        phonemes_indices = {possible_phonemes[i]:i for i in range(possibles_phonemes_number)}
        bigrams = np.zeros((possibles_phonemes_number, possibles_phonemes_number), dtype=int)
        previous_phonemes_counter = np.zeros((possibles_phonemes_number), dtype=int)

        for _, alignment in extended_alignment_dataset:
            previous_phoneme = alignment[0]
            for i in range(len(alignment)):
                current_phoneme = alignment[i]
                bigrams[phonemes_indices[current_phoneme]][phonemes_indices[previous_phoneme]] += 1
                previous_phonemes_counter[phonemes_indices[previous_phoneme]] += 1
                previous_phoneme = current_phoneme

        if wo_diag:
            np.fill_diagonal(bigrams, 0) # Zeroes the diagonal values
        previous_phonemes_counter[previous_phonemes_counter == 0] = 1 # Replace the zeros of the previous phonemes number by one to avoid dividing by zeros
        bigrams = normalize(bigrams / previous_phonemes_counter, axis=1, norm='l1')
        round_bigrams = np.around(bigrams.copy(), decimals=2)

        fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.matshow(round_bigrams)
        ax.set_xticks(np.arange(possibles_phonemes_number))
        ax.set_yticks(np.arange(possibles_phonemes_number))
        ax.set_xticklabels(possible_phonemes)
        ax.set_yticklabels(possible_phonemes)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

        for i in range(possibles_phonemes_number):
            for j in range(possibles_phonemes_number):
                text = ax.text(j, i, round_bigrams[i, j], ha='center', va='center', color='w')

        output_path = self._results_path + os.sep + 'vctk_{}_groundtruth_bigrams_{}{}ms'.format(
            self._alignment_subset, 'wo_diag_' if wo_diag else '',
            int(desired_time_interval * 1000))

        fig.tight_layout()
        fig.savefig(output_path + '.png')
        plt.close(fig)

        with open(output_path + '.npy', 'wb') as f:
            np.save(f, bigrams)

    def compute_groundtruth_phonemes_frequency(self, wo_diag=True):
        ConsoleLogger.status('Computing groundtruth phonemes frequency of VCTK val dataset...')

        alignments_dic = None

        groundtruth_alignments_path = self._results_path + os.sep + \
            'vctk_{}_groundtruth_alignments.pickle'.format(self._alignment_subset)
        with open(groundtruth_alignments_path, 'rb') as f:
            alignments_dic = pickle.load(f)

        desired_time_interval = alignments_dic['desired_time_interval']
        phonemes_counter = alignments_dic['phonemes_counter']
        total_phonemes_apparations = alignments_dic['total_phonemes_apparations']
        
        phonemes_frequency = dict()
        for key, value in phonemes_counter.items():
            phonemes_frequency[key] = value * 100 / total_phonemes_apparations

        phonemes_frequency_sorted_keys = sorted(phonemes_frequency, key=phonemes_frequency.get, reverse=True)
        values = [phonemes_frequency[key] for key in phonemes_frequency_sorted_keys]

        # TODO: add title
        fig, ax = plt.subplots(figsize=(20, 1))
        ax.bar(phonemes_frequency_sorted_keys, values)
        fig.savefig(self._results_path + os.sep + 'vctk_{}_groundtruth_phonemes_frequency_{}ms.png'.format(
            self._alignment_subset, int(desired_time_interval * 1000)), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def compute_groundtruth_average_phonemes_number(self):
        alignments_dic = None
        groundtruth_alignments_path = self._results_path + os.sep + \
            'vctk_{}_groundtruth_alignments.pickle'.format(self._alignment_subset)
        with open(groundtruth_alignments_path, 'rb') as f:
            alignments_dic = pickle.load(f)

        extended_alignment_dataset = alignments_dic['extended_alignment_dataset']

        phonemes_number = list()
        for _, alignment in extended_alignment_dataset:
            phonemes_number.append(len(np.unique(alignment)))
        ConsoleLogger.success('The average number of phonemes per alignment for {} alignments is: {}'.format(
            len(extended_alignment_dataset), np.mean(round(phonemes_number, 2))))

    def compute_empirical_alignments(self):
        ConsoleLogger.status('Computing empirical alignments of VCTK val dataset...')

        all_alignments = list()
        encodings_counter = dict()
        desired_time_interval = 0.01
        data_length = self._configuration['length'] / self._configuration['sampling_rate']
        desired_time_scale = np.arange(data_length * 100) * desired_time_interval
        total_indices_apparations = 0
        loader = self._data_stream.training_loader if self._alignment_subset == 'train' else self._data_stream.validation_loader

        with tqdm(loader) as bar:
            for data in bar:
                valid_originals = data['input_features'].to(self._device).permute(0, 2, 1).contiguous().float()
                speaker_ids = data['speaker_id'].to(self._device)
                wav_filenames = data['wav_filename']

                speaker_id = wav_filenames[0][0].split('/')[-2]

                if speaker_id not in os.listdir(self._vctk.raw_folder + os.sep + 'VCTK-Corpus' + os.sep + 'phonemes'):
                    # TODO: log the missing folders
                    continue

                z = self._model.encoder(valid_originals)
                z = self._model.pre_vq_conv(z)
                _, quantized, _, encodings, _, encoding_indices, _, \
                    _, _, _, _ = self._model.vq(z, compute_distances_if_possible=False)
                valid_reconstructions = self._model.decoder(quantized, self._data_stream.speaker_dic, speaker_ids)
                B = valid_reconstructions.size(0)
                T = encoding_indices.size(0)

                encoding_indices = encoding_indices.view(B, -1).detach().cpu().numpy()
                extended_time_scale = np.arange(T) * (data_length / T)
                min_time = 0.0

                for i in range(len(valid_reconstructions)):
                    wav_filename = wav_filenames[0][i]
                    utterence_key = wav_filename.split('/')[-1].replace('.wav', '')
                    all_alignments.append((utterence_key, encoding_indices[i]))
                    total_indices_apparations += len(encoding_indices[i])
                    for index in encoding_indices[i]:
                        str_index = str(index)
                        if str_index not in encodings_counter:
                            encodings_counter[str_index] = 0
                        encodings_counter[str_index] += 1

        empirical_alignments_path = self._results_path + os.sep + self._experiment_name + \
            '_vctk_{}_empirical_alignments.pickle'.format(self._alignment_subset)
        with open(empirical_alignments_path, 'wb') as f:
            pickle.dump({
                'all_alignments': all_alignments,
                'encodings_counter': encodings_counter,
                'desired_time_interval': desired_time_interval,
                'total_indices_apparations': total_indices_apparations,
                'num_embeddings': self._configuration['num_embeddings']
            }, f)

    def compute_empirical_bigrams_matrix(self, wo_diag=True):
        ConsoleLogger.status('Computing empirical bigrams matrix of VCTK val dataset {}...'.format(
            'without diagonal' if wo_diag else 'with diagonal'
        ))

        alignments_dic = None
        empirical_alignments_path = self._results_path + os.sep + self._experiment_name + \
            '_vctk_{}_empirical_alignments.pickle'.format(self._alignment_subset)
        with open(empirical_alignments_path, 'rb') as f:
            alignments_dic = pickle.load(f)

        all_alignments = alignments_dic['all_alignments']
        encodings_counter = alignments_dic['encodings_counter']
        desired_time_interval = alignments_dic['desired_time_interval']
        total_indices_apparations = alignments_dic['total_indices_apparations']
        num_embeddings = alignments_dic['num_embeddings']

        if num_embeddings > 100:
            ConsoleLogger.warn('Stopping the computation of empirical bigrams matrix because the embedding number ({}) is huge'.format(num_embeddings))
            return

        bigrams = np.zeros((num_embeddings, num_embeddings), dtype=int)
        previous_index_counter = np.zeros((num_embeddings), dtype=int)

        for _, alignment in all_alignments:
            previous_encoding_index = alignment[0]
            for i in range(len(alignment)):
                current_encoding_index = alignment[i]
                bigrams[current_encoding_index][previous_encoding_index] += 1
                previous_index_counter[previous_encoding_index] += 1
                previous_encoding_index = current_encoding_index

        if wo_diag:
            np.fill_diagonal(bigrams, 0) # Zeroes the diagonal values
        previous_index_counter[previous_index_counter == 0] = 1 # Replace the zeros of the previous phonemes number by one to avoid dividing by zeros
        bigrams = normalize(bigrams / previous_index_counter, axis=1, norm='l1')
        round_bigrams = np.around(bigrams.copy(), decimals=2)

        fig, ax = plt.subplots(figsize=(20, 20))

        im = ax.matshow(round_bigrams)
        ax.set_xticks(np.arange(num_embeddings))
        ax.set_yticks(np.arange(num_embeddings))
        ax.set_xticklabels(np.arange(num_embeddings))
        ax.set_yticklabels(np.arange(num_embeddings))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

        for i in range(num_embeddings):
            for j in range(num_embeddings):
                text = ax.text(j, i, round_bigrams[i, j], ha='center', va='center', color='w')

        output_path = self._results_path + os.sep + self._experiment_name + '_vctk_{}_empirical_bigrams_{}{}ms'.format(
            self._alignment_subset, 'wo_diag_' if wo_diag else '', int(desired_time_interval * 1000))

        fig.tight_layout()
        fig.savefig(output_path + '.png')
        plt.close(fig)

        with open(output_path + '.npy', 'wb') as f:
            np.save(f, bigrams)

    def comupte_empirical_encodings_frequency(self):
        ConsoleLogger.status('Computing empirical encodings frequency of VCTK val dataset...')

        alignments_dic = None
        empirical_alignments_path = self._results_path + os.sep + self._experiment_name + \
            '_vctk_{}_empirical_alignments.pickle'.format(self._alignment_subset)
        with open(empirical_alignments_path, 'rb') as f:
            alignments_dic = pickle.load(f)

        encodings_counter = alignments_dic['encodings_counter']
        desired_time_interval = alignments_dic['desired_time_interval']
        total_indices_apparations = alignments_dic['total_indices_apparations']

        encodings_frequency = dict()
        for key, value in encodings_counter.items():
            encodings_frequency[key] = value * 100 / total_indices_apparations

        encodings_frequency_sorted_keys = sorted(encodings_frequency, key=encodings_frequency.get, reverse=True)
        values = [encodings_frequency[key] for key in encodings_frequency_sorted_keys]

        # TODO: add title
        fig, ax = plt.subplots(figsize=(20, 1))
        ax.bar(encodings_frequency_sorted_keys, values)
        fig.savefig(self._results_path + os.sep + self._experiment_name + '_vctk_{}_empirical_frequency_{}ms.png'.format(
            self._alignment_subset, int(desired_time_interval * 1000)), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def compute_clustering_metrics(self):
        groundtruth_alignments_dic = None
        groundtruth_alignments_path = self._results_path + os.sep + \
            'vctk_{}_groundtruth_alignments.pickle'.format(self._alignment_subset)
        with open(groundtruth_alignments_path, 'rb') as f:
            groundtruth_alignments_dic = pickle.load(f)

        empirical_alignments_dic = None
        empirical_alignments_path = self._results_path + os.sep + self._experiment_name + \
            '_vctk_{}_empirical_alignments.pickle'.format(self._alignment_subset)
        with open(empirical_alignments_path, 'rb') as f:
            empirical_alignments_dic = pickle.load(f)

        groundtruth_alignments = np.array(groundtruth_alignments_dic['extended_alignment_dataset'])
        possible_phonemes = list(groundtruth_alignments_dic['possible_phonemes'])
        empirical_alignments = np.array(empirical_alignments_dic['all_alignments'])
        phonemes_indices = {possible_phonemes[i]:i for i in range(len(possible_phonemes))}

        ConsoleLogger.status('#{} possible phonemes: {}'.format(len(possible_phonemes), possible_phonemes))
        ConsoleLogger.status('# of raw groundtruth alignments: {}'.format(len(groundtruth_alignments)))
        ConsoleLogger.status('# of raw empirical alignments: {}'.format(len(empirical_alignments)))

        groundtruth_utterance_keys = set()
        final_groundtruth_alignments = list()
        final_empirical_alignments = list()

        alignment_length = ((self._configuration['length'] / self._configuration['sampling_rate']) * 100) / 2

        for (utterence_key, alignment) in groundtruth_alignments:
            if len(alignment) != alignment_length: # FIXME
                ConsoleLogger.error('len(alignment) != alignment_length: {}'.format(len(alignment)))
                continue
            groundtruth_utterance_keys.add(utterence_key)
            final_groundtruth_alignments.append([phonemes_indices[alignment[i]] for i in range(len(alignment))])

        for (utterence_key, alignment) in empirical_alignments:
            if utterence_key in groundtruth_utterance_keys:
                final_empirical_alignments.append(alignment)

        final_groundtruth_alignments = np.asarray(final_groundtruth_alignments)
        final_empirical_alignments = np.asarray(final_empirical_alignments)

        ConsoleLogger.status('Groundtruth alignments shape: {}'.format(final_groundtruth_alignments.shape))
        ConsoleLogger.status('Empirical alignments shape: {}'.format(final_empirical_alignments.shape))

        final_groundtruth_alignments = final_groundtruth_alignments[:-(final_groundtruth_alignments.shape[0] - final_empirical_alignments.shape[0])] \
            if final_groundtruth_alignments.shape[0] - final_empirical_alignments.shape[0] > 0 \
            else final_groundtruth_alignments

        final_empirical_alignments = final_empirical_alignments[:-(final_empirical_alignments.shape[0] - final_groundtruth_alignments.shape[0])] \
            if final_empirical_alignments.shape[0] - final_groundtruth_alignments.shape[0] > 0 \
            else final_empirical_alignments

        ConsoleLogger.status('Groundtruth alignments samples: {}'.format([final_groundtruth_alignments[i] for i in range(2)]))
        ConsoleLogger.status('Empirical alignments samples: {}'.format([final_empirical_alignments[i] for i in range(2)]))

        concatenated_groundtruth_alignments = np.concatenate(final_groundtruth_alignments)
        concatenated_empirical_alignments = np.concatenate(final_empirical_alignments)

        ConsoleLogger.status('Concatenated groundtruth alignments shape: {}'.format(concatenated_groundtruth_alignments.shape))
        ConsoleLogger.status('Concatenated empirical alignments shape: {}'.format(concatenated_empirical_alignments.shape))

        adjusted_rand_score = sklearn.metrics.adjusted_rand_score(concatenated_groundtruth_alignments, concatenated_empirical_alignments)
        adjusted_mutual_info_score = sklearn.metrics.adjusted_mutual_info_score(concatenated_groundtruth_alignments, concatenated_empirical_alignments)
        normalized_mutual_info_score = sklearn.metrics.normalized_mutual_info_score(concatenated_groundtruth_alignments, concatenated_empirical_alignments)

        ConsoleLogger.success('Adjusted rand score: {}'.format(adjusted_rand_score))
        ConsoleLogger.success('Adjusted mututal info score: {}'.format(adjusted_mutual_info_score))
        ConsoleLogger.success('Normalized adjusted mututal info score: {}'.format(normalized_mutual_info_score))

        with open(self._results_path + os.sep + self._experiment_name + '_adjusted_rand_score.npy', 'wb') as f:
            np.save(f, adjusted_rand_score)

        with open(self._results_path + os.sep + self._experiment_name + '_adjusted_mutual_info_score.npy', 'wb') as f:
            np.save(f, adjusted_mutual_info_score)

        with open(self._results_path + os.sep + self._experiment_name + '_normalized_mutual_info_score.npy', 'wb') as f:
            np.save(f, normalized_mutual_info_score)

        ConsoleLogger.success('All scores from cluestering metrics were successfully saved')

    @staticmethod
    def compute_clustering_metrics_evolution(all_experiments_names, result_path):
        possible_metric_names = [
            'adjusted_rand_score',
            'adjusted_mutual_info_score',
            'normalized_mutual_info_score'
        ]

        scores = dict()

        correct_file_paths = list()
        for file in os.listdir(result_path):
            # Check if a known experiment name is present in the current file name
            if sum([1 if experiment_name in file else 0 for experiment_name in all_experiments_names]) == 0:
                continue

            # Check if a known clustering metric is present in the current file name
            possible_metric_found = None
            for possible_metric in possible_metric_names:
                if possible_metric in file:
                    possible_metric_found = possible_metric
                    break
            if possible_metric_found is None:
                continue
            correct_file_paths.append((file, possible_metric_found))

        # Sort the selected file paths by the number of embedding vectors indicated in the file name
        correct_file_paths = sorted(correct_file_paths, key=lambda x: int(x[0].split('_')[0].split('-')[1]))

        for (file, possible_metric_found) in correct_file_paths:
            # Classify the experiment results in the correct place within the dict scores
            current_experiment_name = file.split('_' + possible_metric_found)[0].split('-')[0]
            if current_experiment_name not in scores:
                scores[current_experiment_name] = dict()
            if possible_metric_found not in scores[current_experiment_name]:
                scores[current_experiment_name][possible_metric_found] = (list(), list())
            scores[current_experiment_name][possible_metric_found][0].append(int(file.split('_' + possible_metric_found)[0].split('-')[1]))
            scores[current_experiment_name][possible_metric_found][1].append(float(np.load(result_path + os.sep + file)))

        #print(json.dumps(scores, sort_keys=True, indent=2)) # TODO: log this line instead of printting it

        fig, axs = plt.subplots(
            len(possible_metric_names),
            1,
            figsize=(5 * len(possible_metric_names), 15),
            sharex=True
        )

        cycol = cycle('bgr') # TODO: replace by random color selection with the number of possible metrics

        def underscored_text_to_uppercased(text):
            return ' '.join([word[0].upper() + word[1:] for word in text.replace('_', ' ').split(' ')])

        for current_experiment_name in scores.keys():
            i = 0
            for clustering_metric in scores[current_experiment_name].keys():
                axs[i].plot(scores[current_experiment_name][clustering_metric][0],
                    scores[current_experiment_name][clustering_metric][1], color=next(cycol))
                axs[i].set_ylabel(underscored_text_to_uppercased(clustering_metric), fontsize=15)
                i += 1
        fig.suptitle(
            'Evolution of the clustering metric scores accross different number of embedding vectors',
            fontsize=20
        )
        axs[-1].set_xlabel('Number of embedding vectors', fontsize=15)
        fig.savefig(result_path + os.sep + 'clustering_metrics_evolution.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    @staticmethod
    def check_clustering_metrics_stability_over_seeds(all_experiments_names, result_path):
        possible_metric_names = [
            'adjusted_mutual_info_score',
            'normalized_mutual_info_score'
        ]

        scores = dict()

        correct_file_paths = list()
        for file in os.listdir(result_path):
            # Check if a known experiment name is present in the current file name
            if sum([1 if experiment_name in file else 0 for experiment_name in all_experiments_names]) == 0:
                continue

            # Check if a known clustering metric is present in the current file name
            possible_metric_found = None
            for possible_metric in possible_metric_names:
                if possible_metric in file:
                    possible_metric_found = possible_metric
                    break
            if possible_metric_found is None:
                continue
            correct_file_paths.append((file, possible_metric_found))

        # Sort the selected file paths by the seed number indicated in the file name
        correct_file_paths = sorted(correct_file_paths, key=lambda x: int(x[0].split('_')[0].split('-')[-1].replace('seed', '')))

        seeds = set()
        experiment_names_wo_seed = set()
        for (file, possible_metric_found) in correct_file_paths:
            current_experiment_name = file.split('_' + possible_metric_found)[0]
            seeds.add(current_experiment_name.split('-')[-1].replace('seed', ''))
            current_experiment_name_wo_seed = '-'.join(current_experiment_name.split('-')[:-1])
            experiment_names_wo_seed.add(current_experiment_name_wo_seed)
            if possible_metric_found not in scores:
                scores[possible_metric_found] = dict()
            if current_experiment_name_wo_seed not in scores[possible_metric_found]:
                scores[possible_metric_found][current_experiment_name_wo_seed] = list()
            scores[possible_metric_found][current_experiment_name_wo_seed].append(float(np.load(result_path + os.sep + file)))

        seeds = sorted(seeds)

        fig, axs = plt.subplots(
            len(possible_metric_names),
            1,
            figsize=(5 * len(possible_metric_names), 10),
            sharex=True
        )

        #print(json.dumps(scores, sort_keys=True, indent=2)) # TODO: log this line instead of printting it
        #print(seeds) # TODO: log this line instead of printting it

        def underscored_text_to_uppercased(text):
            return ' '.join([word[0].upper() + word[1:] for word in text.replace('_', ' ').split(' ')])

        def autolabel(ax, rects, xpos='center', round_height=True):
            """
            Attach a text label above each bar in *rects*, displaying its height.

            *xpos* indicates which side to place the text w.r.t. the center of
            the bar. It can be one of the following {'center', 'right', 'left'}.
            Source: https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
            """

            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0, 'right': 1, 'left': -1}

            for rect in rects:
                height = round(rect.get_height(), 3) if round_height else rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(offset[xpos]*3, 3),  # use 3 points offset
                            textcoords="offset points",  # in both directions
                            ha=ha[xpos], va='bottom')

        i = 0
        bar_rects = list()
        width = 0.35
        for metric_name in list(scores.keys()):
            width_step = width / 2
            for experiment_name in scores[metric_name]:
                indices = np.arange(len(seeds))
                bar_rects.append((axs[i], axs[i].bar(
                    x=indices + width_step,
                    height=scores[metric_name][experiment_name],
                    width=width,
                    label=experiment_name
                )))
                width_step = -width_step
            axs[i].set_ylabel(underscored_text_to_uppercased(metric_name), fontsize=10)
            i += 1

        fig.suptitle(
            'Evolution of the clustering metric scores accross\ndifferent seed values',
            fontsize=20
        )
        axs[-1].set_xticks(np.arange(len(seeds)))
        axs[-1].set_xticklabels(list(seeds))
        axs[-1].autoscale_view()

        for ax, bar_rect in bar_rects:
            autolabel(ax, bar_rect)

        fig.legend(np.unique(axs[-1].get_legend_handles_labels()[1]))
        fig.savefig(result_path + os.sep + 'clustering_metrics_accross_seeds.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def print_biggest_adjusted_scores(self):
        groundtruth_alignments_dic = None
        groundtruth_alignments_path = self._results_path + os.sep + \
            'vctk_{}_groundtruth_alignments.pickle'.format(self._alignment_subset)
        with open(groundtruth_alignments_path, 'rb') as f:
            groundtruth_alignments_dic = pickle.load(f)

        empirical_alignments_dic = None
        empirical_alignments_path = self._results_path + os.sep + self._experiment_name + \
            '_vctk_{}_empirical_alignments.pickle'.format(self._alignment_subset)
        with open(empirical_alignments_path, 'rb') as f:
            empirical_alignments_dic = pickle.load(f)

        groundtruth_alignments = np.array(groundtruth_alignments_dic['extended_alignment_dataset'])
        possible_phonemes = list(groundtruth_alignments_dic['possible_phonemes'])
        empirical_alignments = np.array(empirical_alignments_dic['all_alignments'])
        phonemes_indices = {possible_phonemes[i]:i for i in range(len(possible_phonemes))}

        ConsoleLogger.status('#{} possible phonemes: {}'.format(len(possible_phonemes), possible_phonemes))
        ConsoleLogger.status('# of raw groundtruth alignments: {}'.format(len(groundtruth_alignments)))
        ConsoleLogger.status('# of raw empirical alignments: {}'.format(len(empirical_alignments)))

        groundtruth_utterance_keys = set()
        final_groundtruth_alignments = list()
        final_empirical_alignments = list()

        alignment_length = ((self._configuration['length'] / self._configuration['sampling_rate']) * 100) / 2

        for (utterence_key, alignment) in groundtruth_alignments:
            if len(alignment) != alignment_length: # FIXME
                continue
            groundtruth_utterance_keys.add(utterence_key)
            final_groundtruth_alignments.append([phonemes_indices[alignment[i]] for i in range(len(alignment))])

        for (utterence_key, alignment) in empirical_alignments:
            if utterence_key in groundtruth_utterance_keys:
                final_empirical_alignments.append(alignment)

        final_groundtruth_alignments = np.asarray(final_groundtruth_alignments)
        final_empirical_alignments = np.asarray(final_empirical_alignments)

        ConsoleLogger.status('Groundtruth alignments shape: {}'.format(final_groundtruth_alignments.shape))
        ConsoleLogger.status('Empirical alignments shape: {}'.format(final_empirical_alignments.shape))

        ConsoleLogger.status('Groundtruth alignments samples: {}'.format([final_groundtruth_alignments[i] for i in range(2)]))
        ConsoleLogger.status('Empirical alignments samples: {}'.format([final_empirical_alignments[i] for i in range(2)]))

        biggest_adjusted_rand_score = 0.0
        biggest_adjusted_mutual_info_score = 0.0
        biggest_adjusted_rand_score_index = 0
        biggest_adjusted_mutual_info_score_index = 0
        i = 0
        with tqdm(final_groundtruth_alignments) as bar:
            for groundtruth_alignment in bar:
                empirical_alignment = final_empirical_alignments[i]
                adjusted_rand_score = sklearn.metrics.adjusted_rand_score(groundtruth_alignment, empirical_alignment)
                adjusted_mutual_info_score = sklearn.metrics.adjusted_mutual_info_score(groundtruth_alignment, empirical_alignment)
                if adjusted_rand_score > biggest_adjusted_rand_score:
                    biggest_adjusted_rand_score = adjusted_rand_score
                    biggest_adjusted_rand_score_index = i
                if adjusted_mutual_info_score > biggest_adjusted_mutual_info_score:
                    biggest_adjusted_mutual_info_score = adjusted_mutual_info_score
                    biggest_adjusted_mutual_info_score_index = i
                bar.update(1)
                i += 1

        ConsoleLogger.status('Biggest adjusted rand score: {} at index {}'.format(
            biggest_adjusted_rand_score, biggest_adjusted_rand_score_index))
        ConsoleLogger.status('Biggest adjusted mutual info score: {} at index {}'.format(
            biggest_adjusted_mutual_info_score, biggest_adjusted_mutual_info_score_index))

        ConsoleLogger.status('Alignments of the biggest adjusted rand score. Groundtruth:{} Empirical:{}'.format(
            final_groundtruth_alignments[biggest_adjusted_rand_score_index], final_empirical_alignments[biggest_adjusted_rand_score_index]))
        ConsoleLogger.status('Alignments of the biggest adjusted mutual info score. Groundtruth:{} Empirical:{}'.format(
            final_groundtruth_alignments[biggest_adjusted_mutual_info_score_index], final_empirical_alignments[biggest_adjusted_mutual_info_score_index]))
