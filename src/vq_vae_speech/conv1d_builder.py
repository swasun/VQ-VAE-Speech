import torch.nn as nn


class Conv1DBuilder(object):

    @staticmethod
    def build(in_channels, out_channels, kernel_size, stride=1, use_kaiming_normal=False):
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        if use_kaiming_normal:
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)
        return conv
