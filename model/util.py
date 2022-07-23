import torch
from torch import nn
from torch.optim import lr_scheduler

from variable.model import ReluTypeEnum, GenDisNormTypeEnum, \
    InitWeightTypeEnum, LearningRatePolicyEnum, MainModelNameEnum
from model.fpn import FPN
from model.generator import Generator
from model.multi_scale_discriminator import MultiScaleDiscriminator
from model.psfrgan_model import PSFRGANModel
from model.fpn_model import FPNModel


def init_weights(
        network,
        # Normal is used in pix2pix and CycleGAN paper but xavier and kaiming work better for some applications.
        init_type=InitWeightTypeEnum.NORMAL,
        init_gain=0.02
):
    def init_function(layer):
        classname = layer.__class__.__name__
        if hasattr(layer, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == InitWeightTypeEnum.NORMAL:
                nn.init.normal_(layer.weight.data, 0.0, init_gain)
            elif init_type == InitWeightTypeEnum.XAVIER:
                nn.init.xavier_normal_(layer.weight.data, gain=init_gain)
            elif init_type == InitWeightTypeEnum.KAIMING:
                nn.init.kaiming_normal_(layer.weight.data, a=0, mode='fan_in')
            elif init_type == InitWeightTypeEnum.ORTHOGONAL:
                nn.init.orthogonal_(layer.weight.data, gain=init_gain)
            else:
                raise Exception(f'Init type is unknown. Init type: {init_type}.')

            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias.data, 0.0)
        # BatchNorm2d weight is not a matrix, only normal distribution can be applied.
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(layer.weight.data, 1.0, init_gain)
            nn.init.constant_(layer.bias.data, 0.0)

    print(f'Initialize network with InitWeightTypeEnum: {init_type}.')
    network.apply(init_function)


def apply_norm(network, weight_norm_type):
    for layer in network.modules():
        if isinstance(layer, nn.Conv2d):
            if weight_norm_type == GenDisNormTypeEnum.SPECTRAL:
                nn.utils.spectral_norm(layer)
            elif weight_norm_type == GenDisNormTypeEnum.WEIGHT:
                nn.utils.weight_norm(layer)
            else:
                pass


def build_fpn(
        config,
        is_train=True,
        in_size=512,
        out_size=512,
        min_feat_size=32,
        relu_type=ReluTypeEnum.LEAKY_RELU,
        weight_path=None
):
    fpn = FPN(
        in_size=in_size,
        out_size=out_size,
        min_feature_size=min_feat_size,
        base_channel=64,
        parsing_channel=19,
        relu_type=relu_type,
        norm_type=config.fpn_norm_type,
        channel_range=[32, 256]
    )

    if not is_train:
        fpn.eval()

    if weight_path is not None:
        fpn.load_state_dict(torch.load(weight_path))

    if len(config.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        fpn.to(config.device)
        fpn = torch.nn.DataParallel(fpn, config.gpu_ids, output_device=config.device)

    return fpn


def build_generator(
        config,
        is_train=True,
        weight_norm_type=None,
        relu_type=ReluTypeEnum.LEAKY_RELU):
    gen = Generator(
        out_channel=3,
        out_size=config.generator_out_size,
        relu_type=relu_type,
        norm_type=config.generator_norm
    )

    apply_norm(gen, weight_norm_type)

    if not is_train:
        gen.eval()
    if len(config.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        gen.to(config.device)
        gen = torch.nn.DataParallel(gen, config.gpu_ids, output_device=config.device)

    return gen


def build_discriminator(
        config,
        in_channel=3,
        is_train=True,
        weight_norm_type=None
):
    disc = MultiScaleDiscriminator(
        in_channel=in_channel,
        base_channel=config.num_discriminator_filter,
        num_layer=config.num_discriminator_layer,
        norm_type=config.discriminator_norm,
        num_discriminator=config.num_discriminator
    )

    apply_norm(disc, weight_norm_type)

    if not is_train:
        disc.eval()

    if len(config.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        disc.to(config.device)
        disc = torch.nn.DataParallel(disc, config.gpu_ids, output_device=config.device)

    init_weights(
        network=disc,
        init_type=InitWeightTypeEnum.NORMAL,
        init_gain=0.02
    )

    return disc


# For 'linear', we keep the same learning rate for the first <config.init_learning_rate_num_epoch> epochs
# and linearly decay the rate to zero over the next <config.decay_next_num_epoch> epochs.
# For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
# See https://pytorch.org/docs/stable/optim.html for more details.
def get_scheduler(optimizer, config):
    if config.learning_rate_policy == LearningRatePolicyEnum.LINEAR:
        def lambda_rule(epoch):
            return 1.0 - max(
                0,
                epoch + config.start_epoch - config.init_learning_rate_num_epoch
            ) / float(config.decay_next_num_epoch + 1)

        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda_rule
        )
    elif config.learning_rate_policy == LearningRatePolicyEnum.STEP:
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=config.gamma_decay_iter,
            gamma=0.1
        )
    elif config.learning_rate_policy == LearningRatePolicyEnum.PLATEAU:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            threshold=0.01,
            patience=5
        )
    elif config.learning_rate_policy == LearningRatePolicyEnum.COSINE:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.init_learning_rate_num_epoch,
            eta_min=0
        )
    else:
        raise Exception(f'Learning rate policy is not supported. Learning rate policy: {config.learning_rate_policy}.')

    return scheduler


def create_model(config):
    if config.main_model_name == MainModelNameEnum.PSFRGAN:
        return PSFRGANModel(config)
    elif config.main_model_name == MainModelNameEnum.FPN:
        return FPNModel(config)
    else:
        raise Exception(f'Main model name is not supported. Main model name: {config.main_model_name}.')