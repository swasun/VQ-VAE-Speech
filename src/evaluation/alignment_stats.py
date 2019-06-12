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
                        interval.mark = interval.mark[:-1] if interval.mark[-1].isdigit() else interval.mark
                        interval.minTime = float(interval.minTime)
                        interval.maxTime = float(interval.maxTime)
                        if interval.maxTime < shifting_time:
                            continue
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
                    """indices = list()
                    index = encoding_indices[i][0]
                    total_indices_apparations += 1
                    for j in range(1, len(extended_time_scale)):
                        max_time = extended_time_scale[j]
                        time_difference = max_time - min_time
                        quotient, remainder = divmod(float(time_difference), desired_time_interval)
                        if 1.0 - remainder == 1.0:
                            remainder = 0.0
                        elif 1.0 - remainder >= 1.0 - desired_time_interval:
                            remainder = desired_time_interval
                        for _ in range(int(quotient)):
                            indices.append(index)
                        if remainder != 0.0:
                            indices.append(index)
                        str_index = str(index)
                        if str_index not in encodings_counter:
                            encodings_counter[str_index] = 0
                        encodings_counter[str_index] += 1
                        min_time = max_time
                        index = encoding_indices[i][j]
                        total_indices_apparations += 1
                    if len(indices) == 0:
                        ConsoleLogger.error('Wow')
                        intput('')
                        break
                    else:
                        all_alignments.append((utterence_key, indices))"""

        with open(self._results_path + os.sep + self._experiment_name + '_vctk_empirical_alignments.pickle', 'wb') as f:
            pickle.dump({
                'all_alignments': all_alignments,
                'encodings_counter': encodings_counter,
                'desired_time_interval': desired_time_interval,
                'total_indices_apparations': total_indices_apparations
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

        bigrams = np.zeros((44, 44), dtype=int)
        previous_index_counter = np.zeros((44), dtype=int)

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
        ax.set_xticks(np.arange(44))
        ax.set_yticks(np.arange(44))
        ax.set_xticklabels(np.arange(44))
        ax.set_yticklabels(np.arange(44))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

        for i in range(44):
            for j in range(44):
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

    def test_clustering(self):
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

        ConsoleLogger.success('Adjusted rand score: {}'.format(sklearn.metrics.adjusted_rand_score(concatenated_groundtruth_alignments, concatenated_empirical_alignments)))
        ConsoleLogger.success('Adjusted mututal info score: {}'.format(sklearn.metrics.adjusted_mutual_info_score(concatenated_groundtruth_alignments, concatenated_empirical_alignments)))
        ConsoleLogger.success('Normalized adjusted mututal info score: {}'.format(sklearn.metrics.normalized_mutual_info_score(concatenated_groundtruth_alignments, concatenated_empirical_alignments)))
