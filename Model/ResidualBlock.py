from torch import nn
from ConvolutionalBlock import ConvolutionalBlock


class ResidualBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel=256):
        super().__init__()
        self.block = nn.Sequential(
            ConvolutionalBlock(in_channel, out_channel, is_activation=True, kernel_size=3, padding=1),
            ConvolutionalBlock(in_channel, out_channel, is_activation=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)
