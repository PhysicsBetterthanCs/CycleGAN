import torch
import torch.nn as nn

from ConvolutionalBlock import ConvolutionalBlock
from ResidualBlock import ResidualBlock


class Generator(nn.Module):
    def __init__(
        self,
        in_channel=3,
    ):
        super().__init__()
        channel = [64, 128, 256, 128, 64, 3]
        self.layers_1 = nn.Sequential(
            nn.Conv2d(in_channel, channel[0], kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(channel[0]),
            nn.ReLU(inplace=True)
        )
        self.layers_2 = nn.ModuleList(
            [ConvolutionalBlock(channel[0], channel[1], kernel_size=3, stride=2, padding=1, is_downsample=True,
                                is_activation=True),
             ConvolutionalBlock(channel[1], channel[2], kernel_size=3, stride=2, padding=1, is_downsample=True,
                                is_activation=True)]
        )
        self.layers_3 = nn.Sequential(
            *[ResidualBlock(channel[2]) for _ in range(9)]
        )
        self.layers_4 = nn.ModuleList(
            [ConvolutionalBlock(channel[2], channel[3], kernel_size=3, stride=2, padding=1, is_downsample=False,
                                is_activation=True),
             ConvolutionalBlock(channel[3], channel[4], kernel_size=3, stride=2, padding=1, is_downsample=False,
                                is_activation=True)]
        )
        self.layers_5 = nn.Sequential(
            nn.Conv2d(channel[4], channel[5], kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        )

    def forward(self, x):
        x = self.layers_1(x)
        for layer in self.layers_2:
            x = layer(x)

        x = self.layers_3(x)

        for layer in self.layers_4:
            x = layer(x)
        return torch.tanh(self.layers_5(x))
