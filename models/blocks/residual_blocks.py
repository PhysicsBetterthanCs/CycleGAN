from torch import nn

from models.layers.residual_layer import ResidualLayer


class ResidualBlock(nn.Module):

    def __init__(self, in_channels,n_blocks=9):
        super(ResidualBlock, self).__init__()
        self.n_blocks = n_blocks
        self.layers = nn.Sequential(
            *[ResidualLayer(in_channels) for _ in range(n_blocks)]
        )

    def forward(self, x):
        return self.layers(x)
