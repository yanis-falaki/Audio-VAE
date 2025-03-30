import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    def __init__(self, latentSize, inputShape, convChannels, convKernels, convStrides, convPadding):
        super(Encoder, self).__init__()
        self.numConvLayers = len(convChannels) - 1

        self.mConvLayers = nn.ModuleList()
        self.mBatchNorms = nn.ModuleList()

        currentConvOutputShape = inputShape

        for i in range(self.numConvLayers):
            self.mConvLayers.append(nn.Conv2d(convChannels[i], convChannels[i+1], convKernels[i], convStrides[i], convPadding[i]))
            self.mBatchNorms.append(nn.BatchNorm2d(convChannels[i+1]))

            currentFeatureMapSize = math.floor((currentConvOutputShape[1] + 2*convPadding[i] - convKernels[i]) / convStrides[i]) + 1
            currentConvOutputShape = torch.Size([convChannels[i+1], currentFeatureMapSize, currentFeatureMapSize])

        self.postConvShape = currentConvOutputShape

    def forward(self, x):
        for i in range (self.numConvLayers):
            x = self.mConvLayers[i](x)
            x = self.mBatchNorms[i](x)
            x = F.relu(x)
        
        return x
