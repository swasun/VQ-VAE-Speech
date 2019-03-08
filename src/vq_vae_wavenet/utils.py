import librosa
import numpy as np
import yaml


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
