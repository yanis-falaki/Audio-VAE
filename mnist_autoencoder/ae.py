import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Autoencoder(nn.Module):
    """
    Autoencoder represents a Deep Convolutional AE Architecture with
    mirrored encoder and decoder components.
    """

    def __init__(self, convChannels, convKernels, convStrides, convPadding):
        super(Autoencoder, self).__init__()

        self.mEncoder = Encoder(convChannels, convKernels, convStrides, convPadding)
        self.mDecoder = Decoder(convChannels, convKernels, convStrides, convPadding)

    def forward(self, x):
        return self.mDecoder(self.mEncoder(x))

