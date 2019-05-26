from error_handling.console_logger import ConsoleLogger
from experiments.device_configuration import DeviceConfiguration
from dataset.vctk_features_stream import VCTKFeaturesStream

import yaml
import pickle
import librosa
import numpy as np
import torch
import random
from tqdm import tqdm
import os


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
    set_deterministic_on(1234)

    ConsoleLogger.status('Loading the configuration file {}...'.format(configuration_file_path))
    configuration = None
    with open(configuration_file_path, 'r') as configuration_file:
        configuration = yaml.load(configuration_file)
    device_configuration = DeviceConfiguration.load_from_configuration(configuration)
    data_stream = VCTKFeaturesStream('../data/vctk', configuration, device_configuration.gpu_ids, device_configuration.use_cuda)

    root_path = '.' + os.sep + '..' + os.sep + 'data' + os.sep + 'vctk' + os.sep + 'preprocessed'

    def process(configuration, loader, output_path):
        bar = tqdm(data_stream.loader)
        for data in bar:
            preprocessed_audio = data['preprocessed_audio'].detach().cpu()[0].numpy().squeeze()
            wav_filename = data['wav_filename'][0][0]
            wav_filename_split = wav_filename.split(os.sep)
            speaker_directory = output_path + os.sep + wav_filename_split[-2]
            os.makedirs(speaker_directory, exist_ok=True)
            output_path = speaker_directory + os.sep + wav_filename_split[-1]
            librosa.output.write_wav(output_path, preprocessed_audio, sr=configuration['sampling_rate'])

    process(configuration, data_stream.training_loader, root_path + os.sep + 'train')
    process(configuration, data_stream.validation_loader, root_path + os.sep + 'val')
