import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    def __init__(
            self,
            in_channel=int,
            out_channel=int,
            kernel_size=int,
            stride=1,
            padding=0,
            is_downsample=True,
            is_activation=True,
            **kwargs
    ):
        super().__init__()
        if is_downsample:
            self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(out_channel),
            )
            if is_activation:
                self.main.append(nn.ReLU(inplace=True))
            else:
                self.main.append(nn.Identity())
        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, **kwargs),
                nn.InstanceNorm2d(out_channel)
            )
            if is_activation:
                self.main.append(nn.ReLU(inplace=True))
            else:
                self.main.append(nn.Identity())

    def forward(self, x):
        return self.main(x)
