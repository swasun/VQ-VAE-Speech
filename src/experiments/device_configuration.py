from error_handling.console_logger import ConsoleLogger

import torch


class DeviceConfiguration(object):

    def __init__(self, use_cuda, device, gpu_ids):
        self._use_cuda = use_cuda
        self._device = device
        self._gpu_ids = gpu_ids
    
    @property
    def use_cuda(self):
        return self._use_cuda

    @property
    def device(self):
        return self._device

    @property
    def gpu_ids(self):
        return self._gpu_ids

    @staticmethod
    def load_from_configuration(configuration):
        use_cuda = configuration['use_cuda'] and torch.cuda.is_available() # Use cuda if specified and available
        default_device = 'cuda' if use_cuda else 'cpu' # Use default cuda device if possible or use the cpu
        device = configuration['use_device'] if configuration['use_device'] is not None else default_device # Use a defined device if specified
        gpu_ids = [i for i in range(torch.cuda.device_count())] if configuration['use_data_parallel'] else [] # Resolve the gpu ids if gpu parallelization is specified

        ConsoleLogger.status('The used device is: {}'.format(device))
        ConsoleLogger.status('The gpu ids are: {}'.format(gpu_ids))

        # Sanity checks
        if not use_cuda and configuration['use_cuda']:
            ConsoleLogger.warn("The configuration file specified use_cuda=True but cuda isn't available")
        if configuration['use_data_parallel'] and len(gpu_ids) < 2:
            ConsoleLogger.warn('The configuration file specified use_data_parallel=True but there is only {} GPU available'.format(len(gpu_ids)))

        return DeviceConfiguration(use_cuda, device, gpu_ids)
