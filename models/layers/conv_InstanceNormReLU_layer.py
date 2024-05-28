from torch import nn


class ConvInstanceNormReLULayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, is_upsample=False,is_relu=True):
        super(ConvInstanceNormReLULayer, self).__init__()

        if not is_upsample:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1)
            self.norm = nn.InstanceNorm2d(out_channels)

        if is_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
