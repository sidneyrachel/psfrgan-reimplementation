import torch
from torch import optim, nn

from model.util import build_fpn, build_generator, build_discriminator
from variable.model import GenDisNormTypeEnum, ModelNameEnum, LossNameEnum, VisualNameEnum
from perceptual_loss_feature import PerceptualLossFeature
from fm_loss import FMLoss
from gan_loss import GANLoss
from perceptual_loss import PerceptualLoss
from region_style_loss import RegionStyleLoss
from util.common import tensor_to_numpy, batch_numpy_to_image, colorize_mask
from model.base_model import BaseModel


class PSFRGANModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)

        self.fpn_model = build_fpn(
            config=config,
            weight_path=config.fpn_pretrained_weight_file
        )
        self.gen_model = build_generator(
            config=config,
            weight_norm_type=GenDisNormTypeEnum.SPECTRAL
        )

        if self.is_train:
            self.disc_model = build_discriminator(
                config=config,
                in_channel=config.discriminator_in_channel,
                weight_norm_type=GenDisNormTypeEnum.SPECTRAL
            )

            self.vgg_model = PerceptualLossFeature(weight_path=config.vgg_pretrained_weight_file).to(config.device)

            if len(config.gpu_ids) > 0:
                self.vgg_model = nn.DataParallel(self.vgg_model, config.gpu_ids, output_device=config.device)

        self.model_names = [ModelNameEnum.GENERATOR]
        self.loss_names = [
            LossNameEnum.PIX,
            LossNameEnum.PERCEPTUAL,
            LossNameEnum.GENERATOR,
            LossNameEnum.FEATURE_MATCHING,
            LossNameEnum.DISCRIMINATOR,
            LossNameEnum.SEMANTIC_STYLE
        ]
        self.visual_names = [
            VisualNameEnum.LOW_RES_IMAGE,
            VisualNameEnum.SUPER_RES_IMAGE,
            VisualNameEnum.HIGH_RES_IMAGE,
            VisualNameEnum.LOW_RES_MASK,
            VisualNameEnum.HIGH_RES_MASK
        ]
        self.fm_weights = [1**x for x in range(config.num_discriminator)]

        if self.is_train:
            self.model_names = [ModelNameEnum.GENERATOR, ModelNameEnum.DISCRIMINATOR]
            self.load_model_names = [ModelNameEnum.GENERATOR, ModelNameEnum.DISCRIMINATOR]

            self.fm_criterion = FMLoss().to(config.device)
            self.gan_criterion = GANLoss(config.gan_mode).to(config.device)
            self.perceptual_criterion = PerceptualLoss()
            self.pix_criterion = nn.L1Loss()
            self.region_style_criterion = RegionStyleLoss()

            self.generator_optimizer = optim.Adam(
                [
                    param for param in self.gen_model.parameters() if param.requires_grad
                ],
                lr=config.generator_learning_rate,
                betas=(config.beta1, 0.999)
            )
            self.discriminator_optimizer = optim.Adam(
                [param for param in self.disc_model.parameters() if param.requires_grad],
                lr=config.discriminator_learning_rate,
                betas=(config.beta1, 0.999)
            )
            self.optimizers = [self.generator_optimizer, self.discriminator_optimizer]

    def eval(self):
        self.gen_model.eval()
        self.fpn_model.eval()

    def load_pretrain_models(self):
        self.fpn_model.eval()
        print(f'Load pretrained LQ face parsing network from {self.config.fpn_pretrained_weight_file}.')

        if len(self.opt.gpu_ids) > 0:
            self.fpn_model.module.load_state_dict(torch.load(self.config.fpn_pretrained_weight_file))
        else:
            self.fpn_model.load_state_dict(torch.load(self.config.fpn_pretrained_weight_file))

        self.gen_model.eval()
        print(f'Load pretrained PSFRGAN from {self.config.psfr_pretrained_weight_file}.')

        if len(self.opt.gpu_ids) > 0:
            self.gen_model.module.load_state_dict(torch.load(self.config.psfr_pretrained_weight_file), strict=False)
        else:
            self.gen_model.load_state_dict(torch.load(self.config.psfr_pretrained_weight_file), strict=False)

    def set_input(self, inp, current_iter):
        self.current_iter = current_iter
        self.low_res_image = inp['lr'].to(self.config.device)
        self.high_res_image = inp['hr'].to(self.config.device)
        self.high_res_mask = inp['mask'].to(self.config.device)

        if self.config.debug:
            print(f'[PSFRGAN] Low res image shape: {self.low_res_image.shape}. '
                  f'High res image shape: {self.high_res_image.shape}.')

    def forward(self):
        with torch.no_grad():
            low_res_mask, _ = self.fpn_model(self.low_res_image)
            self.one_hot_low_res_mask = (low_res_mask == low_res_mask.max(dim=1, keepdim=True)[0]).float().detach()

        if self.config.debug:
            print(f'[PSFRGAN] Low res mask shape: {self.one_hot_low_res_mask.shape}.')

        self.super_res_image = self.gen_model(self.low_res_image, self.one_hot_low_res_mask)

        self.real_disc_results = self.disc_model(
            torch.cat((self.high_res_image, self.high_res_mask), dim=1),
            is_feat_returned=True
        )
        self.fake_disc_results = self.disc_model(
            torch.cat((self.super_res_image.detach(), self.high_res_mask), dim=1),
            is_feat_returned=False
        )
        self.fake_gen_results = self.disc_model(
            torch.cat((self.super_res_image, self.high_res_mask), dim=1),
            is_feat_returned=True
        )

        self.super_res_image_feats = self.vgg_model(self.super_res_image)
        self.high_res_image_feats = self.vgg_model(self.high_res_image)

    def generator_backward(self):
        self.pix_loss = self.pix_criterion(self.super_res_image, self.high_res_image) * self.config.pix_lambda

        # Semantic style loss
        self.ss_loss = self.region_style_criterion(
            self.super_res_image_feats,
            self.high_res_image_feats,
            self.high_res_mask
        ) * self.config.ss_lambda

        self.perceptual_loss = self.perceptual_criterion(
            self.super_res_image_feats,
            self.high_res_image_feats
        ) * self.config.perceptual_lambda

        # Feature matching loss
        fm_loss = 0

        for i, weight in zip(range(self.config.num_discriminator), self.fm_weights):
            fm_loss += self.fm_criterion(self.fake_gen_results[i][1], self.real_disc_results[i][1]) * weight

        self.fm_loss = fm_loss * self.opt.lambda_fm / self.config.num_discriminator

        gen_loss = 0

        for i in range(self.config.num_discriminator):
            gen_loss += self.gan_criterion(
                prediction=self.fake_gen_results[i][0],
                is_target_real=True,
                is_for_discriminator=False
            )
        self.gen_loss = gen_loss * self.config.gen_lambda / self.config.num_discriminator

        total_loss = self.pix_loss + self.perceptual_loss + self.fm_loss + self.gen_loss + self.ss_loss

        total_loss.backward()

    def discriminator_backward(self):
        self.disc_loss = 0

        for i in range(self.config.num_discriminator):
            self.disc_loss += 0.5 * (
                    self.gan_criterion(
                        prediction=self.fake_disc_results[i],  # TODO: Check if this is true
                        is_target_real=False
                    ) +
                    self.gan_criterion(
                        prediction=self.real_disc_results[i][0],
                        is_target_real=True
                    )
            )

        self.disc_loss /= self.config.num_discriminator

        self.disc_loss.backward()

    def optimize_parameters(self):
        # Update generator
        self.generator_optimizer.zero_grad()
        self.generator_backward()
        self.generator_optimizer.step()

        # Update discriminator
        self.discriminator_optimizer.zero_grad()
        self.discriminator_backward()
        self.discriminator_optimizer.step()

    def get_current_visual_images(self, size=512):
        numpy_images = [
            tensor_to_numpy(self.low_res_image),
            tensor_to_numpy(self.super_res_image),
            tensor_to_numpy(self.high_res_image)
        ]

        out_images = [batch_numpy_to_image(numpy_image, size) for numpy_image in numpy_images]

        visual_images = []
        visual_images += out_images
        visual_images.append(colorize_mask(self.one_hot_low_res_mask, size))
        visual_images.append(colorize_mask(self.high_res_mask, size))

        return visual_images
