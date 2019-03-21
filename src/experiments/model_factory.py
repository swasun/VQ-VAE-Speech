from vq_vae_features.features_auto_encoder import FeaturesAutoEncoder
from vq_vae_features.trainer import Trainer as FeaturesTrainer
from error_handling.console_logger import ConsoleLogger
from vq_vae_wavenet.wavenet_auto_encoder import WaveNetAutoEncoder
from vq_vae_wavenet.trainer import Trainer as WaveNetTrainer

from torch import nn
import torch.optim as optim


class ModelFactory(object):

    @staticmethod
    def build(configuration, device_configuration, dataset, wo_trainer=False):
        """
        Create an AutoEncoder model using the specified device,
        and use GPU parallelization if it's specified and possible
        """
        ConsoleLogger.status('Building model...')
        if configuration['decoder_type'] == 'features':
            auto_encoder = FeaturesAutoEncoder(configuration, device_configuration.device).to(device_configuration.device)
            optimizer = optim.Adam(auto_encoder.parameters(), lr=configuration['learning_rate'], amsgrad=True) # Create an Adam optimizer instance
            if wo_trainer:
                trainer = FeaturesTrainer(device_configuration.device, auto_encoder, optimizer, dataset, configuration) # Create a trainer instance
        elif configuration['decoder_type'] == 'wavenet':
            auto_encoder = WaveNetAutoEncoder(configuration, dataset.speaker_dic, device_configuration.device).to(device_configuration.device)
            optimizer = optim.Adam(auto_encoder.parameters(), lr=configuration['learning_rate'], amsgrad=True) # Create an Adam optimizer instance
            if wo_trainer:
                trainer = WaveNetTrainer(device_configuration.device, auto_encoder, optimizer, dataset, configuration)

        auto_encoder = nn.DataParallel(auto_encoder) if configuration['use_data_parallel'] and configuration['use_cuda'] else auto_encoder

        if wo_trainer:
            return auto_encoder

        return auto_encoder, trainer
