 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-Speech.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

import numpy as np
from python_speech_features.base import mfcc, logfbank
from python_speech_features import delta


class SpeechFeatures(object):

    default_rate = 16000
    default_filters_number = 13

    @staticmethod
    def mfcc(signal, rate=default_rate, filters_number=default_filters_number):
        mfcc_features = mfcc(signal, rate, numcep=filters_number)
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
        logfbank_features = logfbank(signal, rate, nfilt=filters_number)
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
    def features_from_name(name, signal, rate=default_rate, filters_number=default_filters_number):
        return getattr(SpeechFeatures, name)(signal, rate, filters_number)
