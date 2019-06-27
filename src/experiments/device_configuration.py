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

from error_handling.console_logger import ConsoleLogger

import torch


class DeviceConfiguration(object):

    def __init__(self, use_cuda, device, gpu_ids, use_data_parallel):
        self._use_cuda = use_cuda
        self._device = device
        self._gpu_ids = gpu_ids
        self._use_data_parallel = use_data_parallel
    
    @property
    def use_cuda(self):
        return self._use_cuda

    @property
    def device(self):
        return self._device

    @property
    def gpu_ids(self):
        return self._gpu_ids
    
    @property
    def use_data_parallel(self):
        return self._use_data_parallel

    @staticmethod
    def load_from_configuration(configuration):
        use_cuda = configuration['use_cuda'] and torch.cuda.is_available() # Use cuda if specified and available
        default_device = 'cuda' if use_cuda else 'cpu' # Use default cuda device if possible or use the cpu
        device = configuration['use_device'] if configuration['use_device'] is not None else default_device # Use a defined device if specified
        gpu_ids = [i for i in range(torch.cuda.device_count())] if configuration['use_data_parallel'] else [0] # Resolve the gpu ids if gpu parallelization is specified
        if configuration['use_device'] and ':' in configuration['use_device']:
            gpu_ids = [int(configuration['use_device'].split(':')[1])]

        use_data_parallel = True if configuration['use_data_parallel'] and use_cuda and len(gpu_ids) > 1 else False

        ConsoleLogger.status('The used device is: {}'.format(device))
        ConsoleLogger.status('The gpu ids are: {}'.format(gpu_ids))

        # Sanity checks
        if not use_cuda and configuration['use_cuda']:
            ConsoleLogger.warn("The configuration file specified use_cuda=True but cuda isn't available")
        if configuration['use_data_parallel'] and len(gpu_ids) < 2:
            ConsoleLogger.warn('The configuration file specified use_data_parallel=True but there is only {} GPU available'.format(len(gpu_ids)))

        return DeviceConfiguration(use_cuda, device, gpu_ids, use_data_parallel)
