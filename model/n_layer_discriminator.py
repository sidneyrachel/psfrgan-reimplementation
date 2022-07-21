from torch import nn
from variable.model import ReluTypeEnum, ScaleTypeEnum
from model.conv_layer import ConvLayer


class NLayerDiscriminator(nn.Module):
    def __init__(
            self,
            in_channel=3,
            base_channel=64,
            max_channel=1024,
            depth=4,
            norm_type=None,
            relu_type=ReluTypeEnum.LEAKY_RELU
    ):
        super().__init__()

        self.norm_type = norm_type
        self.in_channel = in_channel

        models = [ConvLayer(
            in_channel=in_channel,
            out_channel=base_channel,
            norm_type=None,  # Check if this is true
            relu_type=relu_type
        )]

        for i in range(depth):
            in_channel = min(base_channel * 2**i, max_channel)
            out_channel = min(base_channel * 2**(i + 1), max_channel)
            models.append(
                ConvLayer(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    scale=ScaleTypeEnum.DOWN_AVG,
                    norm_type=norm_type,
                    relu_type=relu_type
                )
            )

        self.model = nn.Sequential(*self.models)
        self.conv_out = ConvLayer(
            in_channel=out_channel,
            out_channel=1,
            is_padding_used=False
        )

    def forward(self, inp, is_feat_returned=False):
        ret_feats = []
        outp = inp

        for idx, layer in enumerate(self.model):
            outp = layer(outp)
            ret_feats.append(outp)

        outp = self.conv_out(outp)

        if is_feat_returned:
            return outp, ret_feats
        else:
            return outp
