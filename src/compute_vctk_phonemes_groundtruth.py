import textgrid
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == "__main__":
    root_path = '../data/vctk/raw/VCTK-Corpus/phonemes'
    target_interval = 0.01
    extended_alignment_dataset = list()
    possible_phonemes = set()
    phonemes_counter = dict()
    total_phonemes_apparations = 0

    dir_names = os.listdir(root_path)

    with tqdm(total=len(dir_names)) as bar:
        for dir_name in dir_names:
            bar.set_description(dir_name)
            for file_name in os.listdir(root_path + os.sep + dir_name):
                tg_path = root_path + os.sep + dir_name + os.sep + file_name
                tg = textgrid.TextGrid()
                tg.read(tg_path)
                phonemes = list()
                for interval in tg.tiers[1]:
                    if interval.mark in ['sil', '', '-', "'"]:
                        continue
                    mark = interval.mark
                    mark = mark[:-1] if mark[-1].isdigit() else mark
                    possible_phonemes.add(mark)
                    time_difference = interval.maxTime - interval.minTime
                    quotient, remainder = divmod(float(time_difference), target_interval)
                    if 1.0 - remainder == 1.0:
                        remainder = 0.0
                    elif 1.0 - remainder >= 1.0 - target_interval:
                        remainder = target_interval
                    for i in range(int(quotient)):
                        phonemes.append(mark)
                    if remainder != 0.0:
                        phonemes.append(mark)
                    if mark not in phonemes_counter:
                        phonemes_counter[mark] = 0
                    phonemes_counter[mark] += 1
                    total_phonemes_apparations += 1
                extended_alignment_dataset.append(phonemes)
            bar.update(1)

    possible_phonemes = list(possible_phonemes)
    possibles_phonemes_number = len(possible_phonemes)
    print('List of phonemes: {}'.format(possible_phonemes))
    print('Number of phonemes: {}'.format(possibles_phonemes_number))
    phonemes_indices = {possible_phonemes[i]:i for i in range(len(possible_phonemes))}
    bigrams = np.zeros((possibles_phonemes_number, possibles_phonemes_number), dtype=int)
    previous_phonemes_counter = np.zeros((possibles_phonemes_number), dtype=int)

    for alignment in extended_alignment_dataset:
        previous_phoneme = alignment[0]
        for i in range(len(alignment)):
            current_phoneme = alignment[i]
            bigrams[phonemes_indices[current_phoneme]][phonemes_indices[previous_phoneme]] += 1
            previous_phonemes_counter[phonemes_indices[previous_phoneme]] += 1
            previous_phoneme = current_phoneme

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.matshow(bigrams / previous_phonemes_counter)
    ax.set_xticks(np.arange(possibles_phonemes_number))
    ax.set_yticks(np.arange(possibles_phonemes_number))
    ax.set_xticklabels(possible_phonemes)
    ax.set_yticklabels(possible_phonemes)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

    for i in range(possibles_phonemes_number):
        for j in range(possibles_phonemes_number):
            text = ax.text(j, i, bigrams[i, j], ha='center', va='center', color='w')

    phonemes_frequency = dict()
    for key, value in phonemes_counter.items():
        phonemes_frequency[key] = value * 100 / total_phonemes_apparations
    
    print('Phonemes frequency:')
    for key in sorted(phonemes_frequency, key=phonemes_frequency.get, reverse=True):
        print('{}: {}'.format(key, phonemes_frequency[key]))

    fig.tight_layout()
    fig.savefig('../results/vctk_groundtruth_bigrams.png')
    plt.close(fig)
