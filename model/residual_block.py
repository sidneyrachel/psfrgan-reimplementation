from torch import nn
from variable.model import ReluTypeEnum, NormTypeEnum, ScaleTypeEnum
from model.conv_layer import ConvLayer


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            relu_type=ReluTypeEnum.PRELU,
            norm_type=NormTypeEnum.BN,
            scale=None
    ):
        super(ResidualBlock, self).__init__()

        if scale is None and in_channels == out_channels:
            self.shortcut_function = lambda inp: inp
        else:
            self.shortcut_function = ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                scale=scale
            )

        if scale == ScaleTypeEnum.UP:
            scale_configs = [ScaleTypeEnum.UP, None]
        elif scale == ScaleTypeEnum.DOWN:
            scale_configs = [None, ScaleTypeEnum.DOWN]
        elif scale is None:
            scale_configs = [None, None]
        else:
            raise Exception(f'Scale is not supported. Scale: {scale}.')

        self.conv1 = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            scale=scale_configs[0],
            norm_type=norm_type,
            relu_type=relu_type
        )

        self.conv2 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            scale=scale_configs[1],
            norm_type=norm_type,
            relu_type=None
        )

    def forward(self, inp):
        identity = self.shortcut_function(inp)

        res = self.conv1(inp)
        res = self.conv2(res)

        return identity + res
