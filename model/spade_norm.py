from torch import nn

from variable.model import NormTypeEnum


class SPADENorm(nn.Module):
    def __init__(
            self,
            norm_channel,
            ref_channel,
            norm_type=NormTypeEnum.SPADE.value,
            kernel_size=3
    ):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_channel, affine=False)
        mid_channel = 64

        self.norm_type = norm_type

        if self.norm_type == NormTypeEnum.SPADE.value:
            self.conv1 = nn.Sequential(
                nn.Conv2d(ref_channel, mid_channel, kernel_size, 1, kernel_size // 2),
                nn.LeakyReLU(0.2, True)
            )
            self.gamma_conv = nn.Conv2d(mid_channel, norm_channel, kernel_size, 1, kernel_size // 2)
            self.beta_conv = nn.Conv2d(mid_channel, norm_channel, kernel_size, 1, kernel_size // 2)

    def calculate_gamma_beta(self, inp):
        outp = self.conv1(inp)
        gamma = self.gamma_conv(outp)
        beta = self.beta_conv(outp)

        return gamma, beta

    def forward(self, inp, reference_inp):
        normalized_input = self.param_free_norm(inp)

        if inp.shape[-1] != reference_inp.shape[-1]:
            reference_inp = nn.functional.interpolate(reference_inp, inp.shape[2:], mode='bicubic', align_corners=False)

        if self.norm_type == NormTypeEnum.SPADE.value:
            gamma, beta = self.calculate_gamma_beta(reference_inp)

            return (normalized_input * gamma) + beta
        elif self.norm_type == NormTypeEnum.IN.value:
            return normalized_input
        else:
            raise Exception(f'Norm type is not supported. Norm type: {self.norm_type}.')
