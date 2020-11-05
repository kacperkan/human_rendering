import torch.nn as nn

from blocks import DownsampleBlockStride, UpsampleBlockRender, ResidualBlock


class RenderNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(RenderNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv_in = nn.Conv2d(
            n_channels, n_channels, kernel_size=7, padding=3
        )
        self.down1 = DownsampleBlockStride(n_channels, n_channels * 2)
        self.down2 = DownsampleBlockStride(n_channels * 2, n_channels * 4)
        self.down3 = DownsampleBlockStride(n_channels * 4, n_channels * 8)
        self.residual = ResidualBlock(n_channels * 8)
        self.up1 = UpsampleBlockRender(n_channels * 8, n_channels * 4)
        self.up2 = UpsampleBlockRender(n_channels * 4, n_channels * 2)
        self.up3 = UpsampleBlockRender(n_channels * 2, n_channels)
        self.conv_out = nn.Conv2d(
            n_channels, n_classes, kernel_size=7, padding=3
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        for idx in range(6):
            x = self.residual(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        logits = self.conv_out(x)
        return logits


class PatchDiscriminator(nn.Module):
    def __init__(self, n_channels):
        super(PatchDiscriminator, self).__init__()
        self.down1 = DownsampleBlockStride(
            n_channels, 64, kernel_size=4
        )  # (bs, 128, 128, 64)
        self.down2 = DownsampleBlockStride(
            64, 128, kernel_size=4
        )  # (bs, 64, 64, 128)
        self.down2 = DownsampleBlockStride(
            64, 128, kernel_size=4
        )  # (bs, 64, 64, 128)
        self.down3 = DownsampleBlockStride(
            128, 256, kernel_size=4
        )  # (bs, 32, 32, 256)
        self.zero_pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(256, 512, kernel_size=4)
        self.norm = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(512, 1, kernel_size=4)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.zero_pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.zero_pad(x)
        logits = self.out(x)
        return logits
