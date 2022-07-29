import numpy as np
import torch.nn as nn
import torch
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization


# Spatial size of training images. 
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Alpha hyperparams for loss
alpha1 = 15.0
alpha2 = 1.0
alpha_w = 0.2
num_iter = 5


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 16 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.LeakyReLU(0.1,inplace=True)
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x