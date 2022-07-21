import torch
from torch import nn
import numpy as np

from variable.model import ReluTypeEnum, NormTypeEnum


class PSFRGenerator(nn.Module):
    def __init__(
            self,
            out_channel,
            out_size=512,
            min_feat_size=16,
            relu_type=ReluTypeEnum.RELU,
            channel_range=[32, 1024],
            norm_type=NormTypeEnum.SPADE
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
        head = [
            nn.Conv2d(head_ch, head_ch, kernel_size=3, padding=1),
            SPADEResBlock(head_ch, head_ch, ref_ch, relu_type, norm_type),
        ]

        body = []
        for i in range(up_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch // 2)
            body += [
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                    SPADEResBlock(cout, cout, ref_ch, relu_type, norm_type)
                )
            ]
            head_ch = head_ch // 2

        self.img_out = nn.Conv2d(ch_clip(head_ch), output_nc, kernel_size=3, padding=1)

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward_spade(self, net, x, ref):
        for m in net:
            x = self.forward_spade_m(m, x, ref)
        return x

    def forward_spade_m(self, m, x, ref):
        if isinstance(m, SPADENorm) or isinstance(m, SPADEResBlock):
            x = m(x, ref)
        else:
            x = m(x)
        return x

    def forward(self, x, ref):
        b, c, h, w = x.shape
        const_input = self.const_input.repeat(b, 1, 1, 1)
        ref_input = torch.cat((x, ref), dim=1)

        feat = self.forward_spade(self.head, const_input, ref_input)

        for idx, m in enumerate(self.body):
            feat = self.forward_spade(m, feat, ref_input)

        out_img = self.img_out(feat)

        return out_img
