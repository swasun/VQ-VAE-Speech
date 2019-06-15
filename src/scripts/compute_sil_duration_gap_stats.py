from error_handling.console_logger import ConsoleLogger
from experiments.device_configuration import DeviceConfiguration
from dataset.vctk_features_stream import VCTKFeaturesStream

import librosa
import numpy as np
import os
import textgrid
import torch
import yaml
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import pickle


def set_deterministic_on(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_wav(filename, sampling_rate, res_type, top_db):
    raw_audio, _ = librosa.load(filename, sampling_rate, res_type=res_type)
    trimmed_audio, trimming_indices = librosa.effects.trim(raw_audio, top_db=top_db)
    trimmed_audio /= np.abs(trimmed_audio).max()
    trimmed_audio = trimmed_audio.astype(np.float32)
    return raw_audio, trimmed_audio, trimming_indices

if __name__ == "__main__":
    configuration_file_path = '../configurations/vctk_features.yaml'
    set_deterministic_on(1234)

    ConsoleLogger.status('Loading the configuration file {}...'.format(configuration_file_path))
    configuration = None
    with open(configuration_file_path, 'r') as configuration_file:
        configuration = yaml.load(configuration_file, Loader=yaml.FullLoader)
    device_configuration = DeviceConfiguration.load_from_configuration(configuration)
    data_stream = VCTKFeaturesStream('../data/vctk', configuration, device_configuration.gpu_ids, device_configuration.use_cuda)

    res_type = 'kaiser_fast'
    top_db = 20
    N = 0
    audio_filenames = list()
    original_shifting_times = list()
    sil_duration_gaps = list()
    beginning_trimmed_times = list()
    detected_sil_durations = list()

    with tqdm(data_stream.validation_loader) as bar:
        for features in bar:
            audio_filename = features['wav_filename'][0][0]
            shifting_time = features['shifting_time'].item()
            sampling_rate = features['sampling_rate'].item()

            raw_audio, trimmed_audio, trimming_indices = load_wav(audio_filename, sampling_rate, res_type, top_db)

            beginning_trimmed_time = trimming_indices[0] / sampling_rate

            groundtruth_alignment_path = '../data/vctk/raw/VCTK-Corpus/phonemes' + os.sep + \
                os.sep.join(audio_filename.split(os.sep)[-2:]).replace('.wav', '.TextGrid')

            if not os.path.isfile(groundtruth_alignment_path):
                #ConsoleLogger.error('{} file not found'.format(groundtruth_alignment_path))
                continue

            tg = textgrid.TextGrid()
            tg.read(groundtruth_alignment_path)
            detected_sil_duration = 0.0
            for interval in tg.tiers[1]:
                if interval.mark != 'sil':
                    break
                detected_sil_duration += float(interval.maxTime) - float(interval.minTime)

            sil_duration_gaps.append(abs(beginning_trimmed_time - detected_sil_duration))
            audio_filenames.append(audio_filename)
            original_shifting_times.append(shifting_time)
            beginning_trimmed_times.append(beginning_trimmed_time)
            detected_sil_durations.append(detected_sil_duration)
            N += 1

    mean_sil_duration_gaps = sum(sil_duration_gaps) / N

    fig, ax = plt.subplots()
    ax.plot(np.arange(N), sil_duration_gaps)
    ax.set_title('Silence duration gap between montreal alignments and\nlibrosa loading with sil thresh at 20db')
    ax.axhline(y=mean_sil_duration_gaps, xmin=0.0, xmax=1.0, color='r')

    yt = ax.get_yticks() 
    yt = np.append(yt, mean_sil_duration_gaps)
    ax.set_yticks(yt)

    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Number of audio samples')
    ax.set_ylim(bottom=0)
    fig.savefig('sil_duration_gaps.png')
    plt.close(fig)

    ConsoleLogger.success('mean sil duration gap: {}'.format(mean_sil_duration_gaps))

    with open('../results/sil_duration_gap_stats.pickle', 'wb') as file:
        pickle.dump({
            'sil_duration_gaps': sil_duration_gaps,
            'audio_filenames': audio_filenames,
            'original_shifting_times': original_shifting_times,
            'beginning_trimmed_times': beginning_trimmed_times,
            'detected_sil_durations': detected_sil_durations,
            'mean_sil_duration_gaps': mean_sil_duration_gaps
        }, file)
