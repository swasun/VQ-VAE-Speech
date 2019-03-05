import torch.nn as nn


class Jitter(nn.Module):

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self._probability = probability

    def forward(self, inputs):
        return inputs
