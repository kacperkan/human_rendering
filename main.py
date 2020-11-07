import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import click
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as tv_transforms

from common import CONFIG
from datasets import DeepFashionDataModule
from feature_net import FeatureNet
from losses import VGGLoss, adversarial_loss, inpainting_loss
from render_net import PatchDiscriminator, RenderNet
from vgg19 import Vgg19
from textures import MapDensePoseTexModule


class HumanRendering(pl.LightningModule):
    def __init__(self, tex_res: int = 256):
        super().__init__()
        self.feature_net = FeatureNet(3, 16)
        self.render_net = RenderNet(16, 3)
        self.discriminators = nn.ModuleList(
            [
                PatchDiscriminator(3),
                PatchDiscriminator(3),
                PatchDiscriminator(3),
            ]
        )
        self.mapper = MapDensePoseTexModule(tex_res)

        self.vgg19 = Vgg19(requires_grad=False).eval()
        self.vgg_loss = VGGLoss()

        self.disc_total_loss = 0
        self.gen_total_loss = 0
        self.gen_losses = []
        self.disc_losses = []

    def forward(
        self, input_texture: torch.Tensor, target_iuv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_out = self.feature_net(input_texture)
        textured_target = self.mapper(feature_out, target_iuv)
        render_out = self.render_net(textured_target)

        return feature_out, textured_target, render_out

    def training_step(self, train_batch, batch_nb, optimizer_idx):
        texture = train_batch["texture"]
        feature_out, textured_target, render_out = self(
            texture, train_batch["iuv"]
        )
        save_image(feature_out[[0]], "image.png")

        # train generator
        if optimizer_idx == 0:
            loss_inpainting = inpainting_loss(
                feature_out, input_batch, target_batch
            )
            loss_adversarial = adversarial_loss(
                self.discriminators,
                target_batch,
                render_out,
                is_discriminator=False,
            )
            loss_perceptual = self.vgg_loss(
                render_out, target_batch, self.vgg19
            )
            total_loss = loss_inpainting + loss_adversarial + loss_perceptual
            # print(f'\nGen loss: {total_loss}')
            self.gen_total_loss = total_loss
            return total_loss

        # train discriminator
        if optimizer_idx == 1:
            loss_adversarial = adversarial_loss(
                self.discriminators,
                target_batch,
                render_out,
                is_discriminator=True,
            )
            # print(
            #     f"\nDisc loss: {self.disc_total_loss}, {self.gen_total_loss}"
            # )
            self.gen_losses.append(self.gen_total_loss)
            self.disc_losses.append(self.disc_total_loss)
            plt.figure()
            plt.plot(self.gen_losses, label="generator", color="orange")
            plt.plot(self.disc_losses, label="discriminator", color="blue")
            plt.legend()
            plt.savefig("fig.jpg")
            plt.close()
            self.disc_total_loss = 0
            self.disc_total_loss += loss_adversarial
            return loss_adversarial

    def validation_step(self, valid_batch, *args, **kwargs):
        texture = valid_batch["texture"]
        feature_out, textured_target, render_out = self(
            texture, valid_batch["iuv"]
        )
        import ipdb

        ipdb.set_trace()

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0.99

        opt_gen = torch.optim.Adam(
            list(self.feature_net.parameters())
            + list(self.render_net.parameters()),
            lr=lr,
            betas=(b1, b2),
        )
        opt_disc = torch.optim.Adam(
            list(self.discriminators[0].parameters())
            + list(self.discriminators[1].parameters())
            + list(self.discriminators[2].parameters()),
            lr=lr,
            betas=(b1, b2),
        )
        return [opt_gen, opt_disc], []


@click.command()
@click.argument("train_path", type=str)
@click.argument("valid_path", type=str)
def train(train_path: str, valid_path: str):
    model = HumanRendering()
    train_config = CONFIG["training"]

    data_module = DeepFashionDataModule(
        train_path,
        valid_path,
        batch_size=train_config["batch_size"],
        num_workers=0,
        texture_transforms=tv_transforms.Compose([tv_transforms.ToTensor()]),
        iuv_transforms=None,
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=train_config["max_epochs"],
        default_root_dir=Path("models") / "humanrendering",
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
