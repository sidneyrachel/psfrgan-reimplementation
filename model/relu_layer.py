from torch import nn
from variable.model import ReluTypeEnum


class ReluLayer(nn.Module):
    def __init__(self, num_channel, relu_type=ReluTypeEnum.RELU.value):
        super(ReluLayer, self).__init__()

        if relu_type == ReluTypeEnum.RELU.value:
            self.func = nn.ReLU(True)
        elif relu_type == ReluTypeEnum.LEAKY_RELU.value:
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == ReluTypeEnum.PRELU.value:
            self.func = nn.PReLU(num_channel)
        elif relu_type == ReluTypeEnum.SELU.value:
            self.func = nn.SELU(True)
        elif relu_type is None:
            self.func = lambda inp: inp * 1.0
        else:
            raise Exception(f'Relu type is not supported. '
                            f'Relu type: {relu_type}.')

    def forward(self, inp):
        return self.func(inp)
