import torch

from model.base_model import BaseModel
from model.util import build_fpn
from util.common import tensor_to_numpy, colorize_mask, batch_numpy_to_image
from variable.model import ModelNameEnum, LossNameEnum, VisualNameEnum


class FPNModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)

        self.loss_names = [LossNameEnum.FPN, LossNameEnum.PIX]
        self.visual_names = [
            VisualNameEnum.LOW_RES_IMAGE,
            VisualNameEnum.SUPER_RES_IMAGE,
            VisualNameEnum.PREDICTED_MASK,
            VisualNameEnum.GROUND_TRUTH_MASK,
            VisualNameEnum.HIGH_RES_IMAGE
        ]
        self.model_names = [ModelNameEnum.FPN]
        self.fpn_model = build_fpn(config)

        if self.is_train:
            self.fpn_criterion = torch.nn.CrossEntropyLoss()
            self.pix_criterion = torch.nn.L1Loss()
            self.optimizer = torch.optim.Adam(
                self.fpn_model.parameters(),
                lr=config.fpn_learning_rate,
                betas=(0.9, 0.999)
            )
            self.optimizers = [self.optimizer]

    def set_input(self, inp, current_iter=None):
        self.low_res_image = inp['lr'].to(self.config.device)
        self.high_res_image = inp['hr'].to(self.config.device)
        self.ground_truth_mask = inp['mask'].to(self.config.device)

        if self.config.debug:
            print(f'[FPN] Low res image shape: {self.low_res_image.shape}. '
                  f'High res image shape: {self.high_res_image.shape}. '
                  f'Ground truth mask shape: {self.ground_truth_mask.shape}.')

    def load_pretrain_models(self):
        self.fpn_model.eval()

        print(f'Load pretrained LQ face parsing network from {self.config.fpn_pretrained_weight_file}.')

        self.fpn_model.load_state_dict(torch.load(self.config.fpn_pretrained_weight_file))

    def forward(self):
        self.predicted_mask, self.super_res_image = self.fpn_model(self.low_res_image)

        if self.config.debug:
            print(f'[FPN] Predicted mask shape: {self.predicted_mask.shape}. '
                  f'Super res image shape: {self.super_res_image.shape}.')

    def backward(self):
        self.fpn_loss = self.fpn_criterion(self.predicted_mask, self.ground_truth_mask) * self.config.fpn_lambda
        self.pix_loss = self.pix_criterion(self.super_res_image, self.high_res_image) * self.config.fpn_pix_lambda

        loss = self.fpn_loss + self.pix_loss
        loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()  # Clear existing gradients
        self.backward()  # Calculate gradients
        self.optimizer.step()

    def get_current_visual_images(self, size=512):
        numpy_images = [
            tensor_to_numpy(self.low_res_image),
            tensor_to_numpy(self.super_res_image),
            tensor_to_numpy(self.high_res_image)
        ]

        out_images = [batch_numpy_to_image(numpy_image, size) for numpy_image in numpy_images]

        visual_images = [
            out_images[0],
            out_images[1],
            colorize_mask(self.predicted_mask),
            colorize_mask(self.ground_truth_mask.unsqueeze(1)),
            out_images[2]
        ]

        return visual_images