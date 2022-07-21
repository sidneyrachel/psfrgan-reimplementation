import torch
from torch import nn

from variable.model import ReluTypeEnum, GenDisNormTypeEnum
from model.fpn import FPN
from model.generator import Generator


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
        norm_type=config.p_norm_type,
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
