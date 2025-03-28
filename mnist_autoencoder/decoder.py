import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, convChannels, convKernels, convStrides, convPadding):
        super(Decoder, self).__init__()
        self.num_layers = len(convChannels) - 1

        self.mDeconvLayers = nn.ModuleList()
        self.mBatchNorms = nn.ModuleList()

        for i in range(self.num_layers - 1, -1, -1):
            self.mDeconvLayers.append(nn.ConvTranspose2d(convChannels[i + 1], convChannels[i], convKernels[i], convStrides[i], convPadding[i]))
            
            if i > 0:
                self.mBatchNorms.append(nn.BatchNorm2d(convChannels[i]))
    
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.mDeconvLayers[i](x)

            if (i < self.num_layers - 1):
                x = self.mBatchNorms[i](x)
                x = F.relu(x)

        return F.sigmoid(x)