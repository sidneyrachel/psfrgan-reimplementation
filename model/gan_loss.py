import torch
from torch import nn

from variable.model import GANModeEnum


# Do not use sigmoid as the last layer of Discriminator.
# LSGAN needs no sigmoid. Vanilla GAN will handle it with BCEWithLogitsLoss.
class GANLoss(nn.Module):
    def __init__(
            self,
            gan_mode,
            target_real_label=1.0,  # Label for real image
            target_fake_label=0.0  # Label for fake image
    ):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode == GANModeEnum.LSGAN.value:
            self.loss = nn.MSELoss()
        elif gan_mode == GANModeEnum.VANILLA.value:
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == GANModeEnum.HINGE.value:
            pass
        elif gan_mode == GANModeEnum.WGANGP.value:
            self.loss = None
        else:
            raise Exception(f'GAN mode is not supported. GAN mode: {gan_mode}.')

    def get_target_tensor(self, prediction, is_target_real):
        if is_target_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

    # Calculate loss given prediction output from Discriminator and ground truth label.
    def __call__(self, prediction, is_target_real, is_for_discriminator=True):
        if self.gan_mode in [GANModeEnum.LSGAN.value, GANModeEnum.VANILLA.value]:
            target_tensor = self.get_target_tensor(prediction, is_target_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == GANModeEnum.HINGE.value:
            if is_for_discriminator:
                if is_target_real:
                    loss = nn.ReLU()(1 - prediction).mean()
                else:
                    loss = nn.ReLU()(1 + prediction).mean()
            else:
                if not is_target_real:
                    raise Exception(f'The generator hinge loss is for real image.')

                loss = -prediction.mean()
        elif self.gan_mode == GANModeEnum.WGANGP.value:
            if is_target_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()

        return loss
