import torch
import torch.nn as nn

from blocks import BiasInitialization, ResidualBlock


class RenderNet(nn.Module, BiasInitialization):
    def __init__(self, n_channels, n_classes):
        super(RenderNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv_in = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_channels, n_channels, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
        )
        self.down1 = self._block(n_channels, n_channels * 2)
        self.down2 = self._block(n_channels * 2, n_channels * 4)
        self.down3 = self._block(n_channels * 4, n_channels * 8)
        self.residual = nn.Sequential(
            *[ResidualBlock(n_channels * 8) for _ in range(6)]
        )
        self.up1 = self._up_block(n_channels * 8, n_channels * 4)
        self.up2 = self._up_block(n_channels * 4, n_channels * 2)
        self.up3 = self._up_block(n_channels * 2, n_channels)
        self.conv_out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_channels, n_classes, 7, 1, 0),
            nn.Tanh(),
        )
        self._init()

    def _block(self, in_features: int, out_features: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, 2, 1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

    def _up_block(self, in_features: int, out_features: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, 3, 2, 1, 1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.residual(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        logits = self.conv_out(x)
        return logits


class Dummy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PatchDiscriminator(nn.Module, BiasInitialization):
    def __init__(self, n_channels):
        super(PatchDiscriminator, self).__init__()
        self.down1 = self._block(n_channels, 64, False)
        self.down2 = self._block(64, 128, True)
        self.down3 = self._block(128, 256, True)
        self.down4 = self._block(256, 512, True)
        self.out = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, padding=1), nn.Sigmoid()
        )

        self._init()

    def _block(
        self, in_channels: int, out_channels: int, use_norm: bool
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.InstanceNorm2d(out_channels) if use_norm else Dummy(),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        logits = self.out(x)
        return logits.mean(dim=(2, 3))
