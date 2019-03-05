import torch.nn as nn


class Jitter(nn.Module):

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self._probability = probability

    def forward(self, inputs):
        """
        Current algorithm:
        for i in range(len(quantized)):
            before = random(0.12)
            # TODO: check if lower/upper bound is respected
            quantized[i] = quantized[i-1] if before else quantized[i+1]
        """

        return inputs
