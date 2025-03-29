import torch.nn as nn
import torch.nn.functional as F
import math

class Decoder(nn.Module):
    def __init__(self, latentSize, preDeconvShape, convChannels, convKernels, convStrides, convPadding):
        super(Decoder, self).__init__()

        self.preDeconvShape = preDeconvShape
        self.mLinear = nn.Linear(latentSize, math.prod(self.preDeconvShape))

        self.numDeconvLayers = len(convChannels) - 1
        self.mDeconvLayers = nn.ModuleList()
        self.mBatchNorms = nn.ModuleList()

        for i in range(self.numDeconvLayers - 1, -1, -1):
            self.mDeconvLayers.append(nn.ConvTranspose2d(convChannels[i + 1], convChannels[i], convKernels[i], convStrides[i], convPadding[i]))
            
            if i > 0:
                self.mBatchNorms.append(nn.BatchNorm2d(convChannels[i]))
    
    def forward(self, x):
        x = self.mLinear(x)
        x = x.view(x.shape[0], *self.preDeconvShape)

        for i in range(self.numDeconvLayers):
            x = self.mDeconvLayers[i](x)

            if (i < self.numDeconvLayers - 1):
                x = self.mBatchNorms[i](x)
                x = F.relu(x)

        return F.sigmoid(x)