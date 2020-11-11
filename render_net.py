import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock as ResnetBlock

from blocks import BiasInitialization


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size=7, padding=3, stride=1
            ),
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, use_bn: bool
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_features, out_features, kernel_size=3, padding=1, stride=2
            ),
            (
                nn.BatchNorm2d(out_features)
                if use_bn
                else nn.InstanceNorm2d(out_features)
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpSampleBlock(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, use_bn: bool
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, 3, 2, 1, 1),
            (
                nn.BatchNorm2d(out_features)
                if use_bn
                else nn.InstanceNorm2d(out_features)
            ),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RenderNet(nn.Module, BiasInitialization):
    def __init__(self, n_channels: int, n_classes: int, use_bn: bool):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_filters = 64

        self.layers = nn.Sequential(
            *[
                ConvBlock(n_channels, self.base_filters, use_bn),
                DownSampleBlock(
                    self.base_filters, self.base_filters * 2, use_bn
                ),
                DownSampleBlock(
                    self.base_filters * 2, self.base_filters * 4, use_bn
                ),
                DownSampleBlock(
                    self.base_filters * 4, self.base_filters * 8, use_bn
                ),
                nn.Sequential(
                    *[
                        ResnetBlock(
                            self.base_filters * 8,
                            self.base_filters * 8,
                            norm_layer=(
                                nn.BatchNorm2d if use_bn else nn.InstanceNorm2d
                            ),
                        )
                        for _ in range(6)
                    ]
                ),
                UpSampleBlock(
                    self.base_filters * 8, self.base_filters * 4, use_bn
                ),
                UpSampleBlock(
                    self.base_filters * 4, self.base_filters * 2, use_bn
                ),
                UpSampleBlock(
                    self.base_filters * 2, self.base_filters, use_bn
                ),
                nn.Sequential(
                    nn.Conv2d(self.base_filters, n_classes, 7, 1, 3),
                    nn.Tanh(),
                ),
            ]
        )
        self._init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PatchDiscriminator(nn.Module, BiasInitialization):
    def __init__(self, n_channels: int):
        super().__init__()
        self.down1 = self._block(n_channels, 64)
        self.down2 = self._block(64, 128)
        self.down3 = self._block(128, 256)
        self.down4 = self._block(256, 512)
        self.out = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(512, 1, kernel_size=4, padding=1)
            ),
        )

        self._init()

    def _block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        logits = self.out(x)
        return logits
