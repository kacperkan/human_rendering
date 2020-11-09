import torch
import torch.nn as nn
import torch.nn.init


class BiasInitialization:
    def _init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
                torch.nn.init.normal_(module.weight, 0.0, 0.02)


class DoubleConv(nn.Module, BiasInitialization):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        with_initial_activation: bool = True,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        ]
        if with_initial_activation:
            layers = [
                nn.InstanceNorm2d(in_channels),
                nn.ReLU(inplace=True),
            ] + layers

        self.double_conv = nn.Sequential(*layers)
        self._init()

    def forward(self, x):
        return self.double_conv(x)


class DownsampleBlockMax(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels), nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DownsampleBlockStride(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.down_conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of
        # channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpsampleBlockRender(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.up(x)


class ResidualBlock(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channel_num),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channel_num),
        )

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        return x
