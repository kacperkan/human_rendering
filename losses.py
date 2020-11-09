import torch
import torch.nn as nn

import torch.nn.functional


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def l1_distance(tensor, tensor_sub):
    return nn.functional.l1_loss(tensor_sub, tensor)


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y, vgg):
        x_vgg, y_vgg = vgg(x), vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(
                x_vgg[i], y_vgg[i].detach()
            )
        return loss


def adversarial_loss(models, real_image, fake_image, is_discriminator):
    loss_function = nn.BCELoss()
    loss = 0
    pool = nn.AvgPool2d(3, 2, 1, count_include_pad=False)
    for idx, model in enumerate(models):
        disc_real_out = None
        if is_discriminator:
            disc_real_out = model(real_image)
        disc_fake_out = model(fake_image)
        if is_discriminator:
            assert disc_real_out is not None
            real_loss = loss_function(
                disc_real_out, torch.ones_like(disc_real_out)
            )
            fake_loss = loss_function(
                disc_fake_out, torch.zeros_like(disc_fake_out)
            )
            loss += (real_loss + fake_loss) / 2
        else:
            fake_loss = loss_function(
                disc_fake_out, torch.ones_like(disc_fake_out)
            )
            loss += fake_loss
        if is_discriminator:
            real_image = pool(real_image)
        fake_image = pool(fake_image)
    return loss / len(models)


def inpainting_loss(
    features: torch.Tensor,
    input_texture: torch.Tensor,
    target_texture: torch.Tensor,
) -> torch.Tensor:
    loss = nn.L1Loss()

    dist_in_gen = loss(
        features[:, :3, :, :],
        input_texture[:, :3, :, :],
    )
    dist_tar_gen = loss(
        features[:, :3, :, :],
        target_texture[:, :3, :, :],
    )
    return dist_in_gen + dist_tar_gen
