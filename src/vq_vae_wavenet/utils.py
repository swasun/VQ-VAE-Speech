import librosa
import numpy as np
import yaml
from python_speech_features.base import mfcc
from python_speech_features import delta


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def load_wav(filename, params):
    raw, _ = librosa.load(filename, params['sr'], res_type=params['res_type'])
    raw, _ = librosa.effects.trim(raw, params['top_db'])
    raw /= np.abs(raw).max()
    raw = raw.astype(np.float32)

    return raw

def mu_law_encode(x, mu = 256):
    x = x.astype(np.float32)
    y = np.sign(x) * np.log(1 + mu * np.abs(x)) / \
        np.log(1 + mu)
    y = np.digitize(y, 2 * np.arange(mu) / mu - 1) - 1
    return y.astype(np.long)

def mu_law_decode(y, mu = 256):
    y = y.astype(np.float32)
    y = 2 * y / mu - 1
    x = np.sign(y) / mu * ((mu) ** np.abs(y) - 1)
    return x.astype(np.float32)

def compute_features_from_inputs(signal, rate=16000):
    mfcc_features = mfcc(signal, rate)
    d_mfcc_features = delta(mfcc_features, 2)
    a_mfcc_features = delta(d_mfcc_features, 2)
    concatenated_features = np.concatenate((
            mfcc_features,
            d_mfcc_features,
            a_mfcc_features
        ),
        axis=1
    )
    """print('signal.shape: {}'.format(signal.shape))
    print('mfcc_features.shape: {}'.format(mfcc_features.shape))
    print('d_mfcc_features.shape: {}'.format(d_mfcc_features.shape))
    print('a_mfcc_features.shape: {}'.format(a_mfcc_features.shape))
    print('concatenated_features.shape: {}'.format(concatenated_features.shape))"""
    return concatenated_features
