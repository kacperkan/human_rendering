import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import VideoDataset
from feature_net import FeatureNet
from losses import (
    SaveOutput,
    adversarial_loss,
    inpainting_loss,
    perceptual_loss,
)
from render_net import PatchDiscriminator, RenderNet


class HumanRendering(pl.LightningModule):
    def __init__(self, data_path, batch_size=64):
        super(HumanRendering, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.feature_net = FeatureNet(3, 16)
        self.render_net = RenderNet(16, 3)
        self.discriminators = [
            PatchDiscriminator(3),
            PatchDiscriminator(3),
            PatchDiscriminator(3),
        ]
        self.vgg19 = models.vgg19(pretrained=True)
        self.hook = SaveOutput()
        hook_handles = []
        for layer in self.vgg19.modules():
            if isinstance(layer, nn.ReLU):
                handle = layer.register_forward_hook(self.hook)
                hook_handles.append(handle)
        self.disc_total_loss = 0
        self.gen_total_loss = 0
        self.gen_losses = []
        self.disc_losses = []

    def forward(self, x):
        feature_out = self.feature_net(x)
        render_out = self.render_net(feature_out)
        return feature_out, render_out

    def training_step(self, train_batch, batch_nb, optimizer_idx):
        input_batch = train_batch[: int(train_batch.shape[0] / 2), :, :, :]
        input_batch = torch.randn_like(input_batch)
        target_batch = train_batch[int(train_batch.shape[0] / 2) :, :, :, :]
        generated_input = input_batch[0, :, :, :].unsqueeze(0)
        generated_image = self.forward(generated_input)[1]
        save_image(generated_image, "image.png")
        # train generator
        if optimizer_idx == 0:
            print("lol")
            feature_out, render_out = self.forward(input_batch)
            loss_inpainting = inpainting_loss(
                feature_out, input_batch, target_batch
            )
            loss_adversarial = adversarial_loss(
                self.discriminators,
                input_batch,
                target_batch,
                is_discriminator=False,
            )
            self.vgg19.forward(target_batch)
            ground_truth_activations = self.hook.outputs
            self.hook.clear()
            self.vgg19.forward(input_batch)
            generated_activations = self.hook.outputs
            self.hook.clear()
            loss_perceptual = perceptual_loss(
                ground_truth_activations, generated_activations
            )
            total_loss = loss_inpainting + loss_adversarial + loss_perceptual
            # print(f'\nGen loss: {total_loss}')
            self.gen_total_loss = total_loss
            return total_loss

        # train discriminator
        if optimizer_idx == 1:
            print("x")
            loss_adversarial = adversarial_loss(
                self.discriminators,
                input_batch,
                target_batch,
                is_discriminator=True,
            )
            print(
                f"\nDisc loss: {self.disc_total_loss}, {self.gen_total_loss}"
            )
            self.gen_losses.append(self.gen_total_loss / 1000)
            self.disc_losses.append(self.disc_total_loss)
            plt.plot(self.gen_losses)
            plt.plot(self.disc_losses)
            plt.savefig("fig.jpg")
            self.disc_total_loss = 0
            self.disc_total_loss += loss_adversarial
            return loss_adversarial

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0

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

    def train_dataloader(self):
        dataset = VideoDataset(self.data_path)
        return DataLoader(dataset, batch_size=self.batch_size)


model = HumanRendering("data/train/", batch_size=16)
trainer = pl.Trainer(max_epochs=100000)
trainer.fit(model)
