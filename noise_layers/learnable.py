import numpy as np
import torch.nn as nn
import torch
class GAN(nn.Module):

    def __init__(self,device):
        super(GAN, self).__init__()
        # 3 input image channel, 16 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1).to(device)
        self.relu = nn.LeakyReLU(0.1,inplace=True).to(device)
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1).to(device)

    def forward(self, noised_and_cover):
        noised_image = self.conv1(noised_and_cover[0])
        noised_image = self.relu(noised_image)
        noised_image = self.conv2(noised_image)

        return noised_image, noised_and_cover[0]