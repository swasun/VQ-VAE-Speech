from dataset.vctk_dataset import VCTKDataset
from error_handling.console_logger import ConsoleLogger

import numpy as np
import librosa
import yaml
import matplotlib.pyplot as plt
import torch
import random


def load_wav(filename, sampling_rate, res_type, top_db):
    raw, original_rate = librosa.load(filename, sampling_rate, res_type=res_type)
    if top_db is not None and top_db > 0:
        raw, trim_indices = librosa.effects.trim(raw, top_db=top_db)
    raw /= np.abs(raw).max()
    raw = raw.astype(np.float32)
    return raw, original_rate

def set_deterministic_on(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    configuration_file_path = '../configurations/vctk_features.yaml'
    filename = 'p276_330.wav'
    set_deterministic_on(1234)

    ConsoleLogger.status('Loading the configuration file {}...'.format(configuration_file_path))
    configuration = None
    with open(configuration_file_path, 'r') as configuration_file:
        configuration = yaml.load(configuration_file)
    ConsoleLogger.status("configuration['top_db']: {}".format(configuration['top_db']))

    raw_audio, _ = load_wav(filename, configuration['sampling_rate'], configuration['res_type'], None)
    trimmed_audio, trimmed_indexes = load_wav(filename, configuration['sampling_rate'], configuration['res_type'], 60)
    ConsoleLogger.status('trimmed_indexes: {}'.format(trimmed_indexes))
    preprocessed_audio = VCTKDataset.preprocessing_raw(trimmed_audio, configuration['length'])

    fig, axs = plt.subplots(3, 1, figsize=(35, 30))

    axs[0].set_title('Waveform of the original speech signal')
    axs[0].plot(np.arange(len(raw_audio)) / float(configuration['sampling_rate']), raw_audio)

    axs[1].set_title('Waveform of the trimmed speech signal')
    axs[1].plot(np.arange(len(trimmed_audio)) / float(configuration['sampling_rate']), trimmed_audio)

    axs[2].set_title('Waveform of the preprocessed and trimmed speech signal')
    axs[2].plot(np.arange(len(preprocessed_audio)) / float(configuration['sampling_rate']), preprocessed_audio)

    plt.savefig('test.png', bbox_inches='tight', pad_inches=0)
