from torch import nn
from variable.model import ReluTypeEnum, NormTypeEnum, ScaleTypeEnum
from model.conv_layer import ConvLayer


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            relu_type=ReluTypeEnum.PRELU.value,
            norm_type=NormTypeEnum.BN.value,
            scale=None
    ):
        super(ResidualBlock, self).__init__()

        if scale is None and in_channel == out_channel:
            self.shortcut_function = lambda inp: inp
        else:
            self.shortcut_function = ConvLayer(
                in_channel=in_channel,
                out_channel=out_channel,
                kernel_size=3,
                scale=scale
            )

        if scale == ScaleTypeEnum.UP.value:
            scale_configs = [ScaleTypeEnum.UP.value, None]
        elif scale == ScaleTypeEnum.DOWN.value:
            scale_configs = [None, ScaleTypeEnum.DOWN.value]
        elif scale is None:
            scale_configs = [None, None]
        else:
            raise Exception(f'Scale is not supported. Scale: {scale}.')

        self.conv_1 = ConvLayer(
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=3,
            scale=scale_configs[0],
            norm_type=norm_type,
            relu_type=relu_type
        )

        self.conv_2 = ConvLayer(
            in_channel=out_channel,
            out_channel=out_channel,
            kernel_size=3,
            scale=scale_configs[1],
            norm_type=norm_type,
            relu_type=None
        )

    def forward(self, inp):
        identity = self.shortcut_function(inp)

        res = self.conv_1(inp)
        res = self.conv_2(res)

        return identity + res
