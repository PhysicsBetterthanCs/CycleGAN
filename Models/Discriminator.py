from torch import nn

class Discriminator(nn.Module):

    def __init__(self, in_channel=3):
        super().__init__()

        channels = [64, 128, 256, 512]

        def ConvInstanceNormLeakyReLUBlock(
                in_channel,
                out_channel,
                normalize=True,
                kernel_size=4,
                stride=2,
                padding=1,
                activation=None
        ):
            layers = nn.ModuleList(
                [nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False if normalize else True)]
            )

            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))

            layers.append(nn.LeakyReLU(0.2, inplace=True) if activation is None else activation)

            return layers

        self.main = nn.Sequential(
            *ConvInstanceNormLeakyReLUBlock(in_channel, channels[0], normalize=False),
            *ConvInstanceNormLeakyReLUBlock(channels[0], channels[1]),
            *ConvInstanceNormLeakyReLUBlock(channels[1], channels[2]),
            *ConvInstanceNormLeakyReLUBlock(channels[2], channels[3], stride=1),
            *ConvInstanceNormLeakyReLUBlock(channels[3], 1, normalize=False, stride=1, activation=nn.Sigmoid())
        )

    def forward(self, x):
        return self.main(x)
