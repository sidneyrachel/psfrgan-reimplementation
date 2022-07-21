from torch import nn
from variable.model import NormTypeEnum
from model.spade_norm import SPADENorm
from model.relu_layer import ReluLayer


class StyleTransformBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            ref_channels,
            relu_type,
            norm_type=NormTypeEnum.SPADE
    ):
        super().__init__()

        mid_channels = min(in_channels, out_channels)
        self.conv_0 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=1
        )
        self.conv_1 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.norm_0 = SPADENorm(
            norm_channels=mid_channels,
            ref_channels=ref_channels,
            norm_type=norm_type
        )
        self.norm_1 = SPADENorm(
            norm_channels=mid_channels,
            ref_channels=ref_channels,
            norm_type=norm_type
        )
        self.relu = ReluLayer(
            num_channels=mid_channels,
            relu_type=relu_type
        )

    def forward(self, inp, mask):
        res = self.conv_0(self.relu(self.norm_0(inp, mask)))
        res = self.conv_1(self.relu(self.norm_1(res, mask)))
        out = inp + res

        return out
