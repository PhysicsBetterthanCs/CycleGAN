from torch import nn

from models.blocks.residual_blocks import ResidualBlock
from models.layers.conv_InstanceNormReLU_layer import ConvInstanceNormReLULayer


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        channels = [64, 128, 256, 128, 64, 3]
        self.lay_init = nn.Sequential(
            nn.ReflectionPad2d(3),
            ConvInstanceNormReLULayer(in_channels=3, out_channels=channels[0], kernel_size=7, stride=1, padding=0),
        )
        self.lay_downsample = nn.Sequential(
            ConvInstanceNormReLULayer(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=2,
                                      padding=1),
            ConvInstanceNormReLULayer(in_channels=channels[1], out_channels=channels[2], kernel_size=3, stride=2,
                                      padding=1),
        )
        self.lay_resnet = ResidualBlock(channels[2])
        self.lay_upsample = nn.Sequential(
            ConvInstanceNormReLULayer(in_channels=channels[2], out_channels=channels[3], kernel_size=3, stride=2,
                                      padding=1, is_upsample=True),
            ConvInstanceNormReLULayer(in_channels=channels[3], out_channels=channels[4], kernel_size=3, stride=2,
                                      padding=1, is_upsample=True),
        )
        self.lay_output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels[4], 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.lay_init(x)
        x = self.lay_downsample(x)
        x = self.lay_resnet(x)
        x = self.lay_upsample(x)
        x = self.lay_output(x)
        return x
