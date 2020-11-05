import torch
import torch.nn as nn
import torch.nn.functional as F


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def l1_distance(tensor, tensor_sub):
    diff_tensor = torch.abs(tensor - tensor_sub)
    return torch.sum(diff_tensor)


def perceptual_loss(ground_truth_activations, generated_activations):
    loss = 0
    for ground_truth_activation, generated_activation in zip(
        ground_truth_activations, generated_activations
    ):
        num_elements = 1
        for dim in ground_truth_activation.size()[1:]:
            num_elements *= dim
        loss += (
            l1_distance(ground_truth_activation, generated_activation)
            / num_elements
        )
    return loss


def adversarial_loss(models, real_image, fake_image, is_discriminator):
    loss_function = nn.BCELoss()
    loss = 0
    activation = nn.Sigmoid()
    for idx, model in enumerate(models):
        if is_discriminator:
            real_image_down = F.interpolate(
                real_image, scale_factor=1 / (2 ** idx), mode="bilinear"
            )
            disc_real_out = activation(model(real_image_down))
        fake_image_down = F.interpolate(
            fake_image, scale_factor=1 / (2 ** idx), mode="bilinear"
        )
        disc_fake_out = activation(model(fake_image_down))
        if is_discriminator:
            if idx == 0:
                print("\n")
                print(torch.mean(disc_real_out[0]))
                print(torch.mean(disc_fake_out[0]))
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
    return loss / len(models)


def inpainting_loss(features, input_texture, target_texture):
    dist_in_gen = l1_distance(
        input_texture[:, :3, :, :], features[:, :3, :, :]
    )
    dist_tar_gen = l1_distance(
        target_texture[:, :3, :, :], features[:, :3, :, :]
    )
    return dist_in_gen + dist_tar_gen
