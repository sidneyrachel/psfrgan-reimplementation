from torch import nn
import numpy as np

from variable.model import ReluTypeEnum, NormTypeEnum, ScaleTypeEnum
from model.conv_layer import ConvLayer
from model.residual_block import ResidualBlock


class FPN(nn.Module):
    def __init__(
            self,
            in_size=128,
            out_size=128,
            min_feature_size=32,
            base_channel=64,
            parsing_channel=19,
            residual_depth=10,
            relu_type=ReluTypeEnum.PRELU.value,
            norm_type=NormTypeEnum.BN.value,
            channel_range=[32, 512]
    ):
        super().__init__()
        self.residual_depth = residual_depth
        min_channel, max_channel = channel_range
        clip_channel_function = lambda channel: max(min_channel, min(channel, max_channel))
        min_feat_size = min(in_size, min_feature_size)

        down_steps = int(np.log2(in_size // min_feat_size))
        up_steps = int(np.log2(out_size // min_feat_size))

        encoders = [ConvLayer(
            in_channel=3,
            out_channel=base_channel
        )]

        head_channel = base_channel

        for i in range(down_steps):
            in_channel, out_channel = clip_channel_function(head_channel), clip_channel_function(head_channel * 2)
            encoders.append(
                ResidualBlock(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    scale=ScaleTypeEnum.DOWN.value,
                    relu_type=relu_type,
                    norm_type=norm_type
                )
            )
            head_channel = head_channel * 2

        bodies = []

        for i in range(residual_depth):
            bodies.append(
                ResidualBlock(
                    in_channel=clip_channel_function(head_channel),
                    out_channel=clip_channel_function(head_channel),
                    relu_type=relu_type,
                    norm_type=norm_type
                )
            )

        decoders = []

        for i in range(up_steps):
            in_channel, out_channel = clip_channel_function(head_channel), clip_channel_function(head_channel // 2)
            decoders.append(
                ResidualBlock(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    scale=ScaleTypeEnum.UP.value,
                    relu_type=relu_type,
                    norm_type=norm_type
                )
            )
            head_channel = head_channel // 2

        self.encoder = nn.Sequential(*encoders)
        self.body = nn.Sequential(*bodies)
        self.decoder = nn.Sequential(*decoders)
        self.out_img_conv = ConvLayer(
            in_channel=clip_channel_function(head_channel),
            out_channel=3
        )
        self.out_mask_conv = ConvLayer(
            in_channel=clip_channel_function(head_channel),
            out_channel=parsing_channel
        )

    def forward(self, inp):
        feat = self.encoder(inp)
        outp = feat + self.body(feat)
        outp = self.decoder(outp)
        image = self.out_img_conv(outp)
        mask = self.out_mask_conv(outp)

        return mask, image
