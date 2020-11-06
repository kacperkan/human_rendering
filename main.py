import multiprocessing as mp

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import VideoDataset
from feature_net import FeatureNet
from losses import VGGLoss, adversarial_loss, inpainting_loss
from render_net import PatchDiscriminator, RenderNet


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class HumanRendering(pl.LightningModule):
    def __init__(self, data_path, batch_size=64):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.feature_net = FeatureNet(3, 16)
        self.render_net = RenderNet(16, 3)
        self.discriminators = nn.ModuleList(
            [
                PatchDiscriminator(3),
                PatchDiscriminator(3),
                PatchDiscriminator(3),
            ]
        )
        self.vgg19 = Vgg19(requires_grad=False).eval()
        self.vgg_loss = VGGLoss()

        self.disc_total_loss = 0
        self.gen_total_loss = 0
        self.gen_losses = []
        self.disc_losses = []

    def forward(self, x):
        feature_out = self.feature_net(x)
        render_out = self.render_net(feature_out)

        return feature_out, render_out

    def training_step(self, train_batch, batch_nb, optimizer_idx):
        device = next(self.parameters()).device

        train_batch = train_batch.to(device)
        input_batch = train_batch[: int(train_batch.shape[0] / 2), :, :, :]
        input_batch = torch.randn_like(input_batch)
        target_batch = train_batch[int(train_batch.shape[0] / 2) :, :, :, :]
        generated_input = input_batch[0, :, :, :].unsqueeze(0)
        feature_out, render_out = self(input_batch)
        with torch.no_grad():
            generated_image = self(generated_input)[1]
            save_image(generated_image, "image.png")
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

    def train_dataloader(self):
        dataset = VideoDataset(self.data_path)
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=mp.cpu_count() - 1
        )


model = HumanRendering("data/train/", batch_size=8)
if torch.cuda.is_available():
    model = model.cuda()

trainer = pl.Trainer(gpus=1, max_epochs=100000)
trainer.fit(model)
