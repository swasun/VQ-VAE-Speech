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

import os
import sys
sys.path.append('..' + os.sep + '..' + os.sep + 'src')

from vq_vae_speech.global_conditioning import GlobalConditioning
from experiments.device_configuration import DeviceConfiguration
from dataset.vctk_speech_stream import VCTKSpeechStream
from error_handling.console_logger import ConsoleLogger

import unittest
import yaml
import torch


class GlobalConditioningTest(unittest.TestCase):

    def test_global_conditioning(self):
        configuration = None
        with open('../../configurations/vctk_features.yaml', 'r') as configuration_file:
            configuration = yaml.load(configuration_file)
        device_configuration = DeviceConfiguration.load_from_configuration(configuration)
        data_stream = VCTKSpeechStream(configuration, device_configuration.gpu_ids, device_configuration.use_cuda)
        (x_enc, x_dec, speaker_id, _, _) = next(iter(data_stream.training_loader))

        ConsoleLogger.status('x_enc.size(): {}'.format(x_enc.size()))
        ConsoleLogger.status('x_dec.size(): {}'.format(x_dec.size()))

        x = x_dec.squeeze(-1)    
        global_conditioning = GlobalConditioning.compute(
            speaker_dic=data_stream.speaker_dic,
            speaker_ids=speaker_id,
            x_one_hot=x,
            expand=False
        )
        self.assertEqual(global_conditioning.size(), torch.Size([1, 128, 1]))
        ConsoleLogger.success('global_conditioning.size(): {}'.format(global_conditioning.size()))

        expanded_global_conditioning = GlobalConditioning.compute(
            speaker_dic=data_stream.speaker_dic,
            speaker_ids=speaker_id,
            x_one_hot=x,
            expand=True
        )
        self.assertEqual(expanded_global_conditioning.size(), torch.Size([1, 128, 7680]))
        ConsoleLogger.success('expanded_global_conditioning.size(): {}'.format(expanded_global_conditioning.size()))


if __name__ == '__main__':
    unittest.main()
