import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Union

class Encoder(nn.Module):
    def __init__(self, 
                 inputShape: Tuple[int, int, int],  # (channels, height, width)
                 convChannels: List[int],
                 convKernels: List[Union[int, Tuple[int, int]]],
                 convStrides: List[Union[int, Tuple[int, int]]],
                 convPadding: List[Union[int, Tuple[int, int]]]):
        super(Encoder, self).__init__()
        self.numConvLayers = len(convChannels) - 1

        self.mConvLayers = nn.ModuleList()
        self.mBatchNorms = nn.ModuleList()

        # Convert all parameters to tuples if they aren't already
        def to_tuple(param: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
            return (param, param) if isinstance(param, int) else param

        currentShape = inputShape  # (channels, height, width)

        for i in range(self.numConvLayers):
            kernel = to_tuple(convKernels[i])
            stride = to_tuple(convStrides[i])
            padding = to_tuple(convPadding[i])

            self.mConvLayers.append(nn.Conv2d(
                in_channels=convChannels[i],
                out_channels=convChannels[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=convPadding[i]
            ))
            self.mBatchNorms.append(nn.BatchNorm2d(convChannels[i+1]))

            # Calculate output size for height and width separately
            out_height = math.floor((currentShape[1] + 2*padding[0] - kernel[0]) / stride[0]) + 1
            out_width = math.floor((currentShape[2] + 2*padding[1] - kernel[1]) / stride[1]) + 1
            
            # Update shape for next iteration
            currentShape = (convChannels[i+1], out_height, out_width)

        self.postConvShape = currentShape

    def forward(self, x):
        for i in range(self.numConvLayers):
            x = self.mConvLayers[i](x)
            x = self.mBatchNorms[i](x)
            x = F.relu(x)
        
        return x
