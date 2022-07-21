from torch import nn


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_ch, base_ch=64, n_layers=3, norm_type='none', relu_type='LeakyReLU', num_D=4):
        super().__init__()

        self.D_pool = nn.ModuleList()
        for i in range(num_D):
            netD = NLayerDiscriminator(input_ch, base_ch, depth=n_layers, norm_type=norm_type, relu_type=relu_type)
            self.D_pool.append(netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input, return_feat=False):
        results = []
        for netd in self.D_pool:
            output = netd(input, return_feat)
            results.append(output)
            # Downsample input
            input = self.downsample(input)
        return results
