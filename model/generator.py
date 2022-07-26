import torch
from torch import nn
import numpy as np

from variable.model import ReluTypeEnum, NormTypeEnum
from model.style_transform_block import StyleTransformBlock
from model.spade_norm import SPADENorm


class Generator(nn.Module):
    def __init__(
            self,
            out_channel,
            out_size=512,
            min_feat_size=16,
            relu_type=ReluTypeEnum.RELU.value,
            channel_range=[32, 1024],
            norm_type=NormTypeEnum.SPADE.value
    ):
        super().__init__()

        min_channel, max_channel = channel_range
        clip_channel_function = lambda channel: max(min_channel, min(channel, max_channel))
        get_channel = lambda size: clip_channel_function(1024 * 16 // size)

        self.const_input = nn.Parameter(
            torch.randn(
                1,
                get_channel(min_feat_size),
                min_feat_size,
                min_feat_size
            )
        )
        up_steps = int(np.log2(out_size // min_feat_size))
        self.up_steps = up_steps

        ref_channel = 19 + 3

        head_channel = get_channel(min_feat_size)
        heads = [
            nn.Conv2d(head_channel, head_channel, kernel_size=3, padding=1),
            StyleTransformBlock(
                in_channel=head_channel,
                out_channel=head_channel,
                ref_channel=ref_channel,
                relu_type=relu_type,
                norm_type=norm_type
            )
        ]

        bodies = []

        for idx in range(up_steps):
            in_ch, out_ch = clip_channel_function(head_channel), clip_channel_function(head_channel // 2)
            bodies.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    StyleTransformBlock(
                        in_channel=out_ch,
                        out_channel=out_ch,
                        ref_channel=ref_channel,
                        relu_type=relu_type,
                        norm_type=norm_type
                    )
                )
            )
            head_channel = head_channel // 2

        self.img_out = nn.Conv2d(
            clip_channel_function(head_channel),
            out_channel,
            kernel_size=3,
            padding=1
        )

        self.head = nn.Sequential(*heads)
        self.body = nn.Sequential(*bodies)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward_spade_layer(self, layer, inp, reference_inp):
        if isinstance(layer, SPADENorm) or isinstance(layer, StyleTransformBlock):
            outp = layer(inp, reference_inp)
        else:
            outp = layer(inp)

        return outp

    def forward_spade(self, layers, inp, reference_inp):
        outp = inp

        for layer in layers:
            outp = self.forward_spade_layer(layer, outp, reference_inp)

        return outp

    def forward(self, inp, mask):
        batch_size, num_channel, height, weight = inp.shape
        const_input = self.const_input.repeat(batch_size, 1, 1, 1)
        reference_input = torch.cat((inp, mask), dim=1)

        feat = self.forward_spade(self.head, const_input, reference_input)

        for idx, layer in enumerate(self.body):
            feat = self.forward_spade(layer, feat, reference_input)

        outp = self.img_out(feat)

        return outp
