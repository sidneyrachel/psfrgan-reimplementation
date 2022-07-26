from torch import nn
import numpy as np
from variable.model import ScaleTypeEnum, NormTypeEnum
from model.relu_layer import ReluLayer
from model.norm_layer import NormLayer


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            scale=None,
            norm_type=None,
            relu_type=None,
            is_padding_used=True,
            is_bias=True
    ):
        super(ConvLayer, self).__init__()
        self.is_padding_used = is_padding_used
        self.norm_type = norm_type
        self.in_channel = in_channel
        self.is_bias = is_bias
        self.scale = scale

        if norm_type == NormTypeEnum.BN.value:
            self.is_bias = False

        stride = 2 if self.scale == ScaleTypeEnum.DOWN.value else 1

        if scale == ScaleTypeEnum.UP.value:
            self.scale_func = lambda inp: nn.functional.interpolate(inp, scale_factor=2, mode='nearest')
        else:
            self.scale_func = lambda inp: inp

        self.reflection_pad = nn.ReflectionPad2d(int(np.ceil((kernel_size - 1.) / 2)))
        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride, bias=self.is_bias)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.relu = ReluLayer(
            num_channel=out_channel,
            relu_type=relu_type
        )
        self.norm = NormLayer(
            num_channel=out_channel,
            norm_type=norm_type
        )

    def forward(self, inp):
        outp = self.scale_func(inp)

        if self.is_padding_used:
            outp = self.reflection_pad(outp)

        outp = self.conv2d(outp)

        if self.scale == ScaleTypeEnum.DOWN_AVG.value:
            outp = self.avgpool(outp)

        outp = self.norm(outp)
        outp = self.relu(outp)

        return outp
