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
    
    def __init__(self, data_stream, vctk, configuration, device, model, results_path, experiment_name):
        self._data_stream = data_stream
        self._vctk = vctk
        self._configuration = configuration
        self._device = device
        self._model = model
        self._results_path = results_path
        self._experiment_name = experiment_name

        self._model.eval()

    def compute_groundtruth_alignments(self):
        ConsoleLogger.status('Computing groundtruth alignments of VCTK val dataset...')

        desired_time_interval = 0.02
        extended_alignment_dataset = list()
        possible_phonemes = set()
        phonemes_counter = dict()
        total_phonemes_apparations = 0
        data_length = self._configuration['length'] / self._configuration['sampling_rate']

        with tqdm(self._data_stream.validation_loader) as bar:
            for data in bar:
                speaker_ids = data['speaker_id'].to(self._device)
                wav_filenames = data['wav_filename']
                shifting_times = data['shifting_time'].to(self._device)
                loader_indices = data['index'].to(self._device)

                speaker_id = wav_filenames[0][0].split('/')[-2]
                if speaker_id not in os.listdir(self._vctk.raw_folder + os.sep + 'VCTK-Corpus' + os.sep + 'phonemes'):
                    # TODO: log the missing folders
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

                    shifting_time = shifting_times[0].detach().cpu().item()
                    target_time_scale = np.arange((data_length / desired_time_interval) + 1) * desired_time_interval + shifting_time
                    tg = textgrid.TextGrid()
                    tg.read(phonemes_alignment_path)
                    if target_time_scale[-1] > tg.tiers[1][-1].maxTime:
                        ConsoleLogger.error('Shifting time error at {}.pickle'.format(loader_indices[i].detach().cpu().item()))
                        continue

                    phonemes = list()
                    current_target_time_index = 0
                    for interval in tg.tiers[1]:
                        if interval.mark in ['', '-', "'"]:
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
                        if len(phonemes) == (data_length / desired_time_interval):
                            break
                    if len(phonemes) == 0:
                        intervals = ['min:{} max:{} mark:{}'.format(interval.minTime, interval.maxTime, interval.mark) for interval in tg.tiers[1]]
                        ConsoleLogger.error('Error - min:{} max:{} shifting:{} target_time_scale: {} intervals: {}'.format(
                            interval.minTime, interval.maxTime, shifting_time, target_time_scale, intervals))
                    else:
                        extended_alignment_dataset.append((utterence_key, phonemes))

        with open(self._results_path + os.sep + 'vctk_groundtruth_alignments.pickle', 'wb') as f:
            pickle.dump({
                'desired_time_interval': desired_time_interval,
                'extended_alignment_dataset': extended_alignment_dataset,
                'possible_phonemes': possible_phonemes,
                'phonemes_counter': phonemes_counter,
                'total_phonemes_apparations': total_phonemes_apparations
            }, f)

    def compute_groundtruth_bigrams_matrix(self, wo_diag=True):
        ConsoleLogger.status('Computing groundtruth bigrams matrix of VCTK val dataset {}...'.format(
            'without diagonal' if wo_diag else 'with diagonal'
        ))

        alignments_dic = None
        with open(self._results_path + os.sep + 'vctk_groundtruth_alignments.pickle', 'rb') as f:
            alignments_dic = pickle.load(f)

        desired_time_interval = alignments_dic['desired_time_interval']
        extended_alignment_dataset = alignments_dic['extended_alignment_dataset']
        possible_phonemes = alignments_dic['possible_phonemes']
        phonemes_counter = alignments_dic['phonemes_counter']
        total_phonemes_apparations = alignments_dic['total_phonemes_apparations']

        possible_phonemes = list(possible_phonemes)
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

        output_path = self._results_path + os.sep + 'vctk_groundtruth_bigrams_{}{}ms'.format(
            'wo_diag_' if wo_diag else '', int(desired_time_interval * 1000))

        fig.tight_layout()
        fig.savefig(output_path + '.png')
        plt.close(fig)

        with open(output_path + '.npy', 'wb') as f:
            np.save(f, bigrams)

    def compute_groundtruth_phonemes_frequency(self, wo_diag=True):
        ConsoleLogger.status('Computing groundtruth phonemes frequency of VCTK val dataset...')

        alignments_dic = None
        with open(self._results_path + os.sep + 'vctk_groundtruth_alignments.pickle', 'rb') as f:
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
        fig.savefig(self._results_path + os.sep + 'vctk_groundtruth_phonemes_frequency_{}ms.png'.format(
            int(desired_time_interval * 1000)), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def compute_empirical_alignments(self):
        ConsoleLogger.status('Computing empirical alignments of VCTK val dataset...')

        all_alignments = list()
        encodings_counter = dict()
        desired_time_interval = 0.01
        data_length = self._configuration['length'] / self._configuration['sampling_rate']
        desired_time_scale = np.arange(data_length * 100) * desired_time_interval
        total_indices_apparations = 0

        with tqdm(self._data_stream.validation_loader) as bar:
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

        with open(self._results_path + os.sep + self._experiment_name + '_vctk_empirical_alignments.pickle', 'wb') as f:
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
        with open(self._results_path + os.sep + self._experiment_name + '_vctk_empirical_alignments.pickle', 'rb') as f:
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

        output_path = self._results_path + os.sep + self._experiment_name + '_vctk_empirical_bigrams_{}{}ms'.format(
            'wo_diag_' if wo_diag else '', int(desired_time_interval * 1000))

        fig.tight_layout()
        fig.savefig(output_path + '.png')
        plt.close(fig)

        with open(output_path + '.npy', 'wb') as f:
            np.save(f, bigrams)

    def comupte_empirical_encodings_frequency(self):
        ConsoleLogger.status('Computing empirical encodings frequency of VCTK val dataset...')

        alignments_dic = None
        with open(self._results_path + os.sep + self._experiment_name + '_vctk_empirical_alignments.pickle', 'rb') as f:
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
        fig.savefig(self._results_path + os.sep + self._experiment_name + '_vctk_empirical_frequency_{}ms.png'.format(
            int(desired_time_interval * 1000)), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def compute_clustering_metrics(self):
        groundtruth_alignments_dic = None
        with open(self._results_path + os.sep + 'vctk_groundtruth_alignments.pickle', 'rb') as f:
            groundtruth_alignments_dic = pickle.load(f)

        empirical_alignments_dic = None
        with open(self._results_path + os.sep + self._experiment_name + '_vctk_empirical_alignments.pickle', 'rb') as f:
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

        for (utterence_key, alignment) in groundtruth_alignments:
            if len(alignment) != 24: # FIXME
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
