import math
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

        self.bottleneck = nn.Linear(math.prod(self.mEncoder.postConvShape), latentSize)
        self.unBottleneck = nn.Linear(latentSize, math.prod(self.mEncoder.postConvShape))

        self.mDecoder = Decoder(convChannels, convKernels, convStrides, convPadding)

        self.latent = None

    def forward(self, x):
        x = self.mEncoder(x)

        x = torch.flatten(x, start_dim=1)
        self.latent = self.bottleneck(x)

        x = self.unBottleneck(self.latent)
        x = x.view(x.shape[0], *self.mEncoder.postConvShape)

        x = self.mDecoder(x)

        return x
    

class VariationalAutoencoder(nn.Module):
    """
    VariationalAutoencoder represents a Deep Convolutional VAE Architecture with
    mirrored encoder and decoder components.
    """

    def __init__(self, latentSize, inputShape, convChannels, convKernels, convStrides, convPadding):
        super(VariationalAutoencoder, self).__init__()

        self.mEncoder = Encoder(latentSize, inputShape, convChannels, convKernels, convStrides, convPadding)

        self.bottleneck = nn.Linear(math.prod(self.mEncoder.postConvShape), latentSize)
        self.unBottleneck = nn.Linear(latentSize, math.prod(self.mEncoder.postConvShape))

        self.mDecoder = Decoder(convChannels, convKernels, convStrides, convPadding)

        self.latent = None

    def forward(self, x):
        x = self.mEncoder(x)

        x = torch.flatten(x, start_dim=1)
        self.latent = self.bottleneck(x)

        x = self.unBottleneck(self.latent)
        x = x.view(x.shape[0], *self.mEncoder.postConvShape)

        x = self.mDecoder(x)

        return x
