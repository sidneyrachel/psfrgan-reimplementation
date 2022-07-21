from enum import Enum


class NormTypeEnum(Enum):
    BN = 1,
    IN = 2,
    GN = 3,
    PIXEL = 4,
    LAYER = 5


class ReluTypeEnum(Enum):
    RELU = 1,
    LEAKY_RELU = 2,
    PRELU = 3,
    SELU = 4


class ScaleTypeEnum(Enum):
    UP = 1,
    DOWN = 2,
    DOWN_AVG = 3
