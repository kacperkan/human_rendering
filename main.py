import multiprocessing as mp
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from functional import seq

import click
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tv_transforms
from torch.utils.tensorboard.writer import SummaryWriter

from common import CONFIG
from datasets import DeepFashionDataModule
from feature_net import FeatureNet
from losses import VGGLoss, adversarial_loss, inpainting_loss
from render_net import PatchDiscriminator, RenderNet
from textures import MapDensePoseTexModule
from vgg19 import Vgg19


def copy2cpu(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class HumanRendering(pl.LightningModule):
    def __init__(
        self, logging_path: str, tex_res: int = 256, use_bn: bool = True
    ):
        super().__init__()
        self.feature_net = FeatureNet(3, 16, use_bn=use_bn)
        self.render_net = RenderNet(16, 3, use_bn=use_bn)
        self.discriminators = nn.ModuleList(
            [
                PatchDiscriminator(6),
                PatchDiscriminator(6),
                PatchDiscriminator(6),
            ]
        )

        self.mapper = MapDensePoseTexModule(tex_res)

        self.vgg19 = Vgg19(requires_grad=False).eval()
        self.vgg_loss = VGGLoss()

        self.disc_total_loss = 0
        self.gen_total_loss = 0

        self.gen_losses: List[float] = []
        self.disc_losses: List[float] = []

        self.train_logger = SummaryWriter(Path(logging_path) / "train")
        self.valid_logger = SummaryWriter(Path(logging_path) / "valid")

    def forward(  # type: ignore
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
        self.train_logger.add_image(
            "render",
            torchvision.utils.make_grid(render_out.add(1).div(2)),
            global_step=self.global_step,
        )
        self.train_logger.add_image(
            "textured_target",
            torchvision.utils.make_grid(textured_target[:, :3].add(1).div(2)),
            global_step=self.global_step,
        )
        self.train_logger.add_image(
            "real",
            torchvision.utils.make_grid(
                train_batch["target_frame"].add(1).div(2)
            ),
            global_step=self.global_step,
        )

        # train generator
        if optimizer_idx == 0:
            loss_inpainting = inpainting_loss(
                feature_out,
                train_batch["texture"],
                train_batch["target_texture"],
            )
            loss_adversarial = adversarial_loss(
                self.discriminators,
                torch.cat(
                    (
                        self.mapper(
                            train_batch["target_texture"][:, :3],
                            train_batch["iuv"],
                        ),
                        train_batch["target_frame"],
                    ),
                    dim=1,
                ),
                torch.cat((textured_target[:, :3], render_out), dim=1),
                is_discriminator=False,
            )
            loss_perceptual = self.vgg_loss(
                render_out, train_batch["target_frame"], self.vgg19
            )
            total_loss = loss_inpainting + loss_adversarial + loss_perceptual

            for name, val in {
                "total": total_loss,
                "adversarial": loss_adversarial,
                "perceptual": loss_perceptual,
                "inpainting": loss_inpainting,
            }.items():
                self.train_logger.add_scalar(
                    f"generator/{name}",
                    val,
                    global_step=self.global_step,
                )
            return {"loss": total_loss}

        # train discriminator
        if optimizer_idx == 1:
            loss_adversarial = adversarial_loss(
                self.discriminators,
                torch.cat(
                    (
                        self.mapper(
                            train_batch["target_texture"][:, :3],
                            train_batch["iuv"],
                        ),
                        train_batch["target_frame"],
                    ),
                    dim=1,
                ),
                torch.cat((textured_target[:, :3], render_out), dim=1),
                is_discriminator=True,
            )

            for name, val in {"adversarial": loss_adversarial}.items():
                self.train_logger.add_scalar(
                    f"discriminator/{name}",
                    loss_adversarial,
                    global_step=self.global_step,
                )
            return {"loss": loss_adversarial}

    def validation_step(self, valid_batch, *args, **kwargs):
        texture = valid_batch["texture"]
        feature_out, textured_target, render_out = self(
            texture, valid_batch["iuv"]
        )

        # train generator
        loss_inpainting = inpainting_loss(
            feature_out,
            valid_batch["texture"],
            valid_batch["target_texture"],
        )
        loss_adversarial_generator = adversarial_loss(
            self.discriminators,
            torch.cat(
                (
                    valid_batch["target_texture"][:, :3],
                    valid_batch["target_frame"],
                ),
                dim=1,
            ),
            torch.cat((textured_target[:, :3], render_out), dim=1),
            is_discriminator=False,
        )
        loss_perceptual = self.vgg_loss(
            render_out, valid_batch["target_frame"], self.vgg19
        )
        total_generator_loss = (
            loss_inpainting + loss_adversarial_generator + loss_perceptual
        )
        loss_adversarial_discriminator = adversarial_loss(
            self.discriminators,
            torch.cat(
                (
                    valid_batch["target_texture"][:, :3],
                    valid_batch["target_frame"],
                ),
                dim=1,
            ),
            torch.cat((textured_target[:, :3], render_out), dim=1),
            is_discriminator=True,
        )
        total_discriminator_loss = loss_adversarial_discriminator
        return {
            "discriminator/total": total_discriminator_loss,
            "generator/total": total_generator_loss,
            "generator/inpainting": loss_inpainting,
            "generator/perceptual": loss_perceptual,
            "textured_target": textured_target[:, :3],
            "renders": render_out,
        }

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> None:
        textured_target = outputs[-1]["textured_target"]
        render_out = outputs[-1]["renders"]

        aggregated: Dict[str, List[np.ndarray]] = defaultdict(list)
        for output in outputs:
            for key, item in output.items():
                if key not in ["renders", "textured_target"]:
                    aggregated[key].append(copy2cpu(item))

        aggregated_dict = {
            key: np.mean(item) for key, item in aggregated.items()
        }
        if self.valid_logger is not None:
            for key, metric in aggregated_dict.items():
                self.valid_logger.add_scalar(
                    key, metric, global_step=self.global_step
                )
            self.valid_logger.add_image(
                "render",
                torchvision.utils.make_grid(render_out.add(1).div(2)),
                global_step=self.global_step,
            )
            self.valid_logger.add_image(
                "textured_target",
                torchvision.utils.make_grid(textured_target.add(1).div(2)),
                global_step=self.global_step,
            )

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


def find_last_checkpoint(path: Path) -> str:
    return (
        seq(path.rglob("*.ckpt"))
        .sorted(key=lambda x: int(x.name.split(".")[0].split("=")[1]))
        .last()
        .as_posix()
    )


@click.command()
@click.argument("train_path", type=str)
@click.argument("valid_path", type=str)
@click.option("--load_pretrained", default=False)
def train(train_path: str, valid_path: str, load_pretrained: bool):
    model_path = Path("models") / "humanrendering"
    if model_path.exists() and not load_pretrained:
        shutil.rmtree(model_path.as_posix())
    model_path.mkdir(exist_ok=True, parents=True)

    train_config = CONFIG["training"]
    model = HumanRendering(
        model_path.as_posix(), use_bn=not train_config["test_run"]
    )
    if load_pretrained:
        model.load_from_checkpoint(find_last_checkpoint(model_path))

    batch_size = (
        train_config["batch_size"] if not train_config["test_run"] else 1
    )
    num_workers = mp.cpu_count() - 1 if not train_config["test_run"] else 0

    data_module = DeepFashionDataModule(
        train_path,
        valid_path,
        batch_size=batch_size,
        num_workers=num_workers,
        texture_transforms=tv_transforms.Compose([tv_transforms.ToTensor()]),
        iuv_transforms=None,
        test_run=train_config["test_run"],
    )

    trainer = pl.Trainer(
        logger=None,
        gpus=1,
        max_epochs=train_config["max_epochs"],
        default_root_dir=model_path,
        num_sanity_val_steps=2,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
