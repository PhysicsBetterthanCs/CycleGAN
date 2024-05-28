from torch import nn


class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
        )

    def forward(self, x):
        return x + self.layer(x)
