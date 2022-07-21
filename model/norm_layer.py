from torch import nn
from variable.model import NormTypeEnum


class NormLayer(nn.Module):
    def __init__(self, num_channel, normalize_shape=None, norm_type=NormTypeEnum.BN):
        super(NormLayer, self).__init__()

        if norm_type == NormTypeEnum.BN:
            self.norm = nn.BatchNorm2d(num_channel, affine=True)
        elif norm_type == NormTypeEnum.IN:
            self.norm = nn.InstanceNorm2d(num_channel, affine=False)
        elif norm_type == NormTypeEnum.GN:
            self.norm = nn.GroupNorm(32, num_channel, affine=True)
        elif norm_type == NormTypeEnum.PIXEL:
            self.norm = lambda inp: nn.functional.normalize(inp, p=2, dim=1)
        elif norm_type == NormTypeEnum.LAYER:
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type is None:
            self.norm = lambda inp: inp * 1.0
        else:
            raise Exception(f'Norm type is not supported. '
                            f'Norm type: {norm_type}.')

    def forward(self, inp):
        return self.norm(inp)
