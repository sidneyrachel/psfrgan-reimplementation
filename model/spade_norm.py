from torch import nn

from variable.model import NormTypeEnum


class SPADENorm:
    def __init__(
            self,
            norm_channel,
            ref_channel,
            norm_type=NormTypeEnum.SPADE,
            kernel_size=3
    ):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_channel, affine=False)
        mid_channel = 64

        self.norm_type = norm_type

        if self.norm_type == NormTypeEnum.SPADE:
            self.conv_1 = nn.Sequential(
                nn.Conv2d(ref_channel, mid_channel, kernel_size, 1, kernel_size // 2),
                nn.LeakyReLU(0.2, True)
            )
            self.gamma_conv = nn.Conv2d(mid_channel, norm_channel, kernel_size, 1, kernel_size // 2)
            self.beta_conv = nn.Conv2d(mid_channel, norm_channel, kernel_size, 1, kernel_size // 2)

    def calculate_gamma_beta(self, inp):
        outp = self.conv_1(inp)
        gamma = self.gamma_conv(outp)
        beta = self.beta_conv(outp)

        return gamma, beta

    def forward(self, inp, mask):
        normalized_input = self.param_free_norm(inp)

        if inp.shape[-1] != mask.shape[-1]:
            mask = nn.functional.interpolate(mask, inp.shape[2:], mode='bicubic', align_corners=False)

        if self.norm_type == NormTypeEnum.SPADE:
            gamma, beta = self.calculate_gamma_beta(mask)

            return (normalized_input * gamma) + beta
        elif self.norm_type == NormTypeEnum.IN:
            return normalized_input
        else:
            raise Exception(f'Norm type is not supported. Norm type: {self.norm_type}.')
