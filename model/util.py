import torch

from variable.model import ReluTypeEnum
from model.fpn_model import FPNModel


def build_fpn_model(
        config,
        in_size=512,
        out_size=512,
        min_feat_size=32,
        relu_type=ReluTypeEnum.LEAKY_RELU,
        weight_path=None
):
    fpn = FPNModel(
        in_size=in_size,
        out_size=out_size,
        min_feature_size=min_feat_size,
        base_channels=64,
        parsing_channels=19,
        relu_type=relu_type,
        norm_type=config.p_norm_type,
        channel_range=[32, 256]
    )

    if not config.is_train:
        fpn.eval()

    if weight_path is not None:
        fpn.load_state_dict(torch.load(weight_path))

    if len(config.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        fpn.to(config.device)
        fpn = torch.nn.DataParallel(fpn, config.gpu_ids, output_device=config.device)

    return fpn
