from vq_vae_wavenet.wavenet_type import WaveNetType
from wavenet_vocoder.builder import wavenet as build_wavenet_model
from flow_wavenet.train import build_model as build_flowavenet_model
from clarinet.train import build_model as build_clarinet_model


class WaveNetFactory(object):
    
    @staticmethod
    def build(wavenet_type):
        if wavenet_type == WaveNetType.WaveNet:
            return build_wavenet_model()
        elif wavenet_type == WaveNetType.FlowWaveNet:
            return build_flowavenet_model()
        elif wavenet_type == WaveNetType.ClariNet:
            return build_clarinet_model()
        raise ValueError('Unsupported WaveNet type')
