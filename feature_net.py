import torch
import torch.nn as nn

from blocks import BiasInitialization


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            (
                nn.BatchNorm2d(out_channels)
                if use_bn
                else nn.InstanceNorm2d(out_channels)
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            (
                nn.BatchNorm2d(out_channels)
                if use_bn
                else nn.InstanceNorm2d(out_channels)
            ),
            nn.ReLU(inplace=True),
        ]

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DownSampleBlock(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, use_bn: bool
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2, 2), ConvBlock(in_features, out_features, use_bn)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpSampleBlock(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, use_bn: bool
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.convo = ConvBlock(in_features, out_features, use_bn)

    def forward(
        self, current_level: torch.Tensor, prev_level: torch.Tensor
    ) -> torch.Tensor:
        upped = self.up(current_level)
        return self.convo(torch.cat((upped, prev_level), dim=1))


class FeatureNet(nn.Module, BiasInitialization):
    def __init__(self, n_channels: int, n_classes: int, use_bn: bool):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = ConvBlock(n_channels, 64, use_bn)
        self.down1 = DownSampleBlock(64, 128, use_bn)
        self.down2 = DownSampleBlock(128, 256, use_bn)
        self.down3 = DownSampleBlock(256, 512, use_bn)
        self.down4 = DownSampleBlock(512, 512, use_bn)
        self.up1 = UpSampleBlock(1024, 256, use_bn)
        self.up2 = UpSampleBlock(512, 128, use_bn)
        self.up3 = UpSampleBlock(256, 64, use_bn)
        self.up4 = UpSampleBlock(128, 64, use_bn)
        self.outc = nn.Sequential(
            ConvBlock(64, 64, use_bn),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self._init()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
