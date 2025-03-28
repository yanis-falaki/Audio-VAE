import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, convChannels, convKernels, convStrides, convPadding):
        super(Encoder, self).__init__()
        self.num_layers = len(convChannels) - 1

        self.mConvLayers = nn.ModuleList()
        self.mBatchNorms = nn.ModuleList()

        for i in range(self.num_layers):
            self.mConvLayers.append(nn.Conv2d(convChannels[i], convChannels[i+1], convKernels[i], convStrides[i], convPadding[i]))

            if (i < self.num_layers - 1):
                self.mBatchNorms.append(nn.BatchNorm2d(convChannels[i+1]))

    def forward(self, x):
        for i in range (self.num_layers):
            x = self.mConvLayers[i](x)

            if (i < self.num_layers - 1):
                x = self.mBatchNorms[i](x)
                x = F.relu(x)
        
        return x
