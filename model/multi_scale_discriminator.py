from torch import nn
from model.n_layer_discriminator import NLayerDiscriminator
from variable.model import ReluTypeEnum


class MultiScaleDiscriminator(nn.Module):
    def __init__(
            self,
            in_channel,
            base_channel=64,
            num_layer=3,
            norm_type=None,
            relu_type=ReluTypeEnum.LEAKY_RELU,
            num_discriminator=4
    ):
        super().__init__()

        self.discriminator_pool = nn.ModuleList()
        for i in range(num_discriminator):
            discriminator = NLayerDiscriminator(
                in_channel=in_channel,
                base_channel=base_channel,
                depth=num_layer,
                norm_type=norm_type,
                relu_type=relu_type
            )

            self.discriminator_pool.append(discriminator)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, inp, is_feat_returned=False):
        results = []

        for discriminator in self.discriminator_pool:
            outp = discriminator(inp, is_feat_returned)
            results.append(outp)

            inp = self.downsample(inp)

        return results
