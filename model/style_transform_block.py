from torch import nn
from variable.model import NormTypeEnum
from model.spade_norm import SPADENorm
from model.relu_layer import ReluLayer


class StyleTransformBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            ref_channel,
            relu_type,
            norm_type=NormTypeEnum.SPADE
    ):
        super().__init__()

        mid_channel = min(in_channel, out_channel)
        self.conv_0 = nn.Conv2d(
            in_channel,
            mid_channel,
            kernel_size=3,
            padding=1
        )
        self.conv_1 = nn.Conv2d(
            mid_channel,
            out_channel,
            kernel_size=3,
            padding=1
        )

        self.norm_0 = SPADENorm(
            norm_channel=mid_channel,
            ref_channel=ref_channel,
            norm_type=norm_type
        )
        self.norm_1 = SPADENorm(
            norm_channel=mid_channel,
            ref_channel=ref_channel,
            norm_type=norm_type
        )
        self.relu = ReluLayer(
            num_channel=mid_channel,
            relu_type=relu_type
        )

    def forward(self, inp, reference_inp):
        res = self.conv_0(self.relu(self.norm_0(inp, reference_inp)))
        res = self.conv_1(self.relu(self.norm_1(res, reference_inp)))
        out = inp + res

        return out
