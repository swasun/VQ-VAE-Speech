import torch.nn as nn
import numpy as np


class Jitter(nn.Module):

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self._probability = probability

    def forward(self, quantized):
        length = quantized.size(2)
        for i in range(length):
            """
            Each latent vector is replace with either of its neighbors with a certain probability
            (0.12 from the paper).
            """
            replace = [True, False][np.random.choice([1, 0], p=[self._probability, 1 - self._probability])]
            if replace:
                if i == 0:
                    neighbor_index = i + 1
                elif i == length - 1:
                    neighbor_index = i - 1
                else:
                    """
                    "We independently sample whether it is to
                    be replaced with the token right after
                    or before it."
                    """
                    neighbor_index = i + np.random.choice([-1, 1], p=[0.5, 0.5])
                quantized[:, i, :] = quantized[:, neighbor_index, :]

        return quantized
