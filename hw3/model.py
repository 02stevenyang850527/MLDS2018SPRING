import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import USE_CUDA


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: 
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, noise_size, cond_size, num_filters, out_channels):
        super(Generator, self).__init__()
        self.embed = nn.Linear(noise_size + cond_size, num_filters * 8 * 4 * 4, bias=False)
        self.upsample = nn.Sequential(
            nn.BatchNorm2d(num_filters * 8),
            
            # Block1: output shape (num_filters * 4, 8, 8)
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(inplace=True),
            
            # Block2: output shape (num_filters * 2, 16, 16)
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(inplace=True),
            
            # Block3: output shape (num_filters, 32, 32)
            nn.ConvTranspose2d(num_filters * 2, num_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),

            # Block4: output shape (out_channels, 64, 64)
            nn.ConvTranspose2d(num_filters, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.cond_size = cond_size
        self.noise_size = noise_size
        self.num_filters = num_filters
        self.out_channels = out_channels

    def forward(self, noises, conds):
        inputs = torch.cat((noises, conds), dim=1)
        inputs = self.embed(inputs)
        inputs = inputs.view(-1, self.num_filters * 8, 4, 4)
        outputs = self.upsample(inputs)
        return outputs


class Discriminator(nn.Module):
    def __init__(self, cond_size, in_channels, num_filters):
        super(Discriminator, self).__init__()
        self.downsample = nn.Sequential(
            # Block1: output shape (num_filters, 32, 32)
            nn.Conv2d(in_channels, num_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # Block2: output shape (num_filters * 2, 16, 16)
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block3: output shape (num_filters * 4, 8, 8)
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block4: output shape (num_filters * 8, 4, 4)
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.hidden2out = nn.Sequential(
            nn.Conv2d(num_filters * 8 + cond_size, num_filters, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, 1, 4, 1),
            nn.Sigmoid()
        )


    def forward(self, imgs, conds):
        hiddens = self.downsample(imgs)
        conds = conds.view(*conds.size(), 1, 1)
        conds = conds.repeat(1, 1, 4, 4)
        hiddens = torch.cat((hiddens, conds), dim=1)
        outputs = self.hidden2out(hiddens)
        outputs = outputs.squeeze()
        return outputs

'''
class Generator(nn.Module):
    def __init__(self, noise_size, cond_size, num_filters, out_channels):
        super(Generator, self).__init__()
        self.embed = nn.Linear(noise_size + cond_size, num_filters * 8 * 4 * 4, bias=False)
        self.upsample = nn.Sequential(
            nn.BatchNorm2d(num_filters * 8),
            
            # Block1: output shape (num_filters * 4, 8, 8)
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_filters * 4),
            
            # Block2: output shape (num_filters * 2, 16, 16)
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_filters * 2),
            
            # Block3: output shape (num_filters, 32, 32)
            nn.ConvTranspose2d(num_filters * 2, num_filters, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_filters),

            # Block4: output shape (out_channels, 64, 64)
            nn.ConvTranspose2d(num_filters, out_channels, 4, 2, 1),
            nn.Tanh()
        )

        self.cond_size = cond_size
        self.noise_size = noise_size
        self.num_filters = num_filters
        self.out_channels = out_channels

    def forward(self, noises, conds):
        inputs = torch.cat((noises, conds), dim=1)
        inputs = self.embed(inputs)
        inputs = inputs.view(-1, self.num_filters * 8, 4, 4)
        outputs = self.upsample(inputs)
        return outputs


class Discriminator(nn.Module):
    def __init__(self, cond_size, in_channels, num_filters):
        super(Discriminator, self).__init__()
        self.downsample = nn.Sequential(
            # Block1: output shape (num_filters, 32, 32)
            nn.Conv2d(in_channels, num_filters, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_filters),

            # Block2: output shape (num_filters * 2, 16, 16)
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_filters * 2),
            
            # Block3: output shape (num_filters * 4, 8, 8)
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_filters * 4),
            
            # Block4: output shape (num_filters * 8, 4, 4)
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_filters * 8)
        )

        self.hidden2out = nn.Sequential(
            nn.Conv2d(num_filters * 8 + cond_size, num_filters, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, 1, 4, 1),
            nn.Sigmoid()
        )


    def forward(self, imgs, conds):
        hiddens = self.downsample(imgs)
        conds = conds.view(*conds.size(), 1, 1)
        conds = conds.repeat(1, 1, 4, 4)
        hiddens = torch.cat((hiddens, conds), dim=1)
        outputs = self.hidden2out(hiddens)
        outputs = outputs.squeeze()
        return outputs
'''
