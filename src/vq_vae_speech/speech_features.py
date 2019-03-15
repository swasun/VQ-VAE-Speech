import numpy as np
from python_speech_features.base import mfcc, logfbank
from python_speech_features import delta


class SpeechFeatures(object):

    default_rate = 16000
    default_filters_number = 13

    @staticmethod
    def mfcc(signal, rate=default_rate, filters_number=default_filters_number):
        mfcc_features = mfcc(signal.cpu(), rate, numcep=filters_number)
        d_mfcc_features = delta(mfcc_features, 2)
        a_mfcc_features = delta(d_mfcc_features, 2)
        concatenated_features = np.concatenate((
                mfcc_features,
                d_mfcc_features,
                a_mfcc_features
            ),
            axis=1
        )
        return concatenated_features

    @staticmethod
    def logfbank(signal, rate=default_rate, filters_number=default_filters_number):
        logfbank_features = logfbank(signal.cpu(), rate, nfilt=filters_number)
        d_logfbank_features = delta(logfbank_features, 2)
        a_logfbank_features = delta(d_logfbank_features, 2)
        concatenated_features = np.concatenate((
                logfbank_features,
                d_logfbank_features,
                a_logfbank_features
            ),
            axis=1
        )
        return concatenated_features

    @staticmethod
    def features_by_name(name, signal, rate=default_rate, filters_number=default_filters_number):
        return getattr(SpeechFeatures, name)(signal, rate, filters_number)
