import torch
from torch import nn
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from PIL import Image

from variable.model import ReluTypeEnum, GenDisNormTypeEnum, \
    InitWeightTypeEnum, LearningRatePolicyEnum
from model.fpn import FPN
from model.generator import Generator
from model.multi_scale_discriminator import MultiScaleDiscriminator


def init_weights(
        network,
        # Normal is used in pix2pix and CycleGAN paper but xavier and kaiming work better for some applications.
        init_type=InitWeightTypeEnum.NORMAL.value,
        init_gain=0.02
):
    def init_function(layer):
        classname = layer.__class__.__name__
        if hasattr(layer, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == InitWeightTypeEnum.NORMAL.value:
                nn.init.normal_(layer.weight.data, 0.0, init_gain)
            elif init_type == InitWeightTypeEnum.XAVIER.value:
                nn.init.xavier_normal_(layer.weight.data, gain=init_gain)
            elif init_type == InitWeightTypeEnum.KAIMING.value:
                nn.init.kaiming_normal_(layer.weight.data, a=0, mode='fan_in')
            elif init_type == InitWeightTypeEnum.ORTHOGONAL.value:
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
            if weight_norm_type == GenDisNormTypeEnum.SPECTRAL.value:
                nn.utils.spectral_norm(layer)
            elif weight_norm_type == GenDisNormTypeEnum.WEIGHT.value:
                nn.utils.weight_norm(layer)
            else:
                pass


def build_fpn(
        config,
        is_train=True,
        in_size=512,
        out_size=512,
        min_feat_size=32,
        relu_type=ReluTypeEnum.LEAKY_RELU.value,
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
        relu_type=ReluTypeEnum.LEAKY_RELU.value):
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
        init_type=InitWeightTypeEnum.NORMAL.value,
        init_gain=0.02
    )

    return disc


# For 'linear', we keep the same learning rate for the first <config.init_learning_rate_num_epoch> epochs
# and linearly decay the rate to zero over the next <config.decay_next_num_epoch> epochs.
# For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
# See https://pytorch.org/docs/stable/optim.html for more details.
def get_scheduler(optimizer, config):
    if config.learning_rate_policy == LearningRatePolicyEnum.LINEAR.value:
        def lambda_rule(epoch):
            return 1.0 - max(
                0,
                epoch + config.start_epoch - config.init_learning_rate_num_epoch
            ) / float(config.decay_next_num_epoch + 1)

        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda_rule
        )
    elif config.learning_rate_policy == LearningRatePolicyEnum.STEP.value:
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=config.gamma_decay_iter,
            gamma=0.1
        )
    elif config.learning_rate_policy == LearningRatePolicyEnum.PLATEAU.value:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            threshold=0.01,
            patience=5
        )
    elif config.learning_rate_policy == LearningRatePolicyEnum.COSINE.value:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.init_learning_rate_num_epoch,
            eta_min=0
        )
    else:
        raise Exception(f'Learning rate policy is not supported. Learning rate policy: {config.learning_rate_policy}.')

    return scheduler


def scale_image_by_width(image, target_width, method=Image.BICUBIC):
    image_width, image_height = image.size

    if image_width == target_width:
        return image

    target_height = int(target_width * image_height / image_width)

    return image.resize((target_width, target_height), method)


def crop_image(image, pos, size):
    image_width, image_height = image.size
    x_pos, y_pos = pos
    target_width = target_height = size

    if image_width > target_width or image_height > target_height:
        return image.crop((x_pos, y_pos, x_pos + target_width, y_pos + target_height))

    return image


def flip_image(image, is_flipped):
    if is_flipped:
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    return image


def print_size_warning(image_width, image_height, width, height):
    if not hasattr(print_size_warning, 'is_printed'):
        print(f'The image size needs to be a multiple of 4. '
              f'Loaded image: {image_width}x{image_height}. '
              f'Loaded image is adjusted to: {width}x{height}.')

        print_size_warning.is_printed = True


def make_image_to_power_2_multiplication(image, base, method=Image.BICUBIC):
    image_width, image_height = image.size

    height = int(round(image_height / base) * base)
    width = int(round(image_width / base) * base)

    if (height == image_height) and (width == image_width):
        return image

    print_size_warning(image_width, image_height, width, height)

    return image.resize((width, height), method)


def compose_transform(
        config,
        params=None,
        grayscale=False,
        method=Image.BICUBIC,
        is_converted=True
):
    transform_actions = []

    if grayscale:
        transform_actions.append(transforms.Grayscale(1))
    if config.preprocess is not None and 'resize' in config.preprocess:
        output_size = [config.scale_size, config.scale_size]
        transform_actions.append(transforms.Resize(output_size, method))
    elif config.preprocess is not None and 'scale_width' in config.preprocess:
        transform_actions.append(transforms.Lambda(lambda image: scale_image_by_width(image, config.scale_size, method)))

    if config.preprocess is not None and 'crop' in config.preprocess:
        if params is None:
            transform_actions.append(transforms.RandomCrop(config.crop_size))
        else:
            if 'crop_size' in params:
                transform_actions.append(transforms.Lambda(
                    lambda image: crop_image(image, params['crop_pos'], params['crop_size'])
                ))
            else:
                transform_actions.append(transforms.Lambda(
                    lambda image: crop_image(image, params['crop_pos'], config.crop_size)
                ))

    if config.preprocess is None:
        transform_actions.append(transforms.Lambda(
            lambda image: make_image_to_power_2_multiplication(image, base=4, method=method)
        ))

    if config.is_flipped:
        if params is None:
            transform_actions.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_actions.append(transforms.Lambda(
                lambda image: flip_image(image, params['flip'])
            ))

    if is_converted:
        transform_actions += [transforms.ToTensor()]

        if grayscale:
            transform_actions += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_actions += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_actions)
