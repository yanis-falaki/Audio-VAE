import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Autoencoder(nn.Module):
    """
    Autoencoder represents a Deep Convolutional AE Architecture with
    mirrored encoder and decoder components.
    """

    def __init__(self, latentSize, inputShape, convChannels, convKernels, convStrides, convPadding):
        super(Autoencoder, self).__init__()

        self.mEncoder = Encoder(latentSize, inputShape, convChannels, convKernels, convStrides, convPadding)
        self.mDecoder = Decoder(latentSize, self.mEncoder.postConvShape, convChannels, convKernels, convStrides, convPadding)
        self.latent = None

    def forward(self, x):
        self.latent = self.mEncoder(x)
        return self.mDecoder(self.latent)

