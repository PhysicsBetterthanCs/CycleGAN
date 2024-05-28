from torch import nn

from models.layers.conv_InstanceNormReLU_layer import ConvInstanceNormReLULayer


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        channels = [64, 128, 256, 512]
        self.layers = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=4, stride=2, padding=1),
            ConvInstanceNormReLULayer(channels[0], channels[1], kernel_size=4, stride=2, padding=1, is_relu=False),
            ConvInstanceNormReLULayer(channels[1], channels[2], kernel_size=4, stride=2, padding=1, is_relu=False),
            ConvInstanceNormReLULayer(channels[2], channels[3], kernel_size=4, stride=1, padding=1, is_relu=False),
            nn.Conv2d(channels[3], 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
