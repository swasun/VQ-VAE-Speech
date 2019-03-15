import numpy as np
from python_speech_features.base import mfcc, logfbank
from python_speech_features import delta


class SpeechFeatures(object):

    @staticmethod
    def mfcc(signal, rate=16000, filters_number=13):
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
    def logfbank(signal, rate=16000, filters_number=13):
        logfbank_features = logfbank(signal.cpu(), rate, nfilt=13)
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
