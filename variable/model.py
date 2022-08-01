from enum import Enum


class NormTypeEnum(Enum):
    BN = 'bn'
    IN = 'in'
    GN = 'gn'
    PIXEL = 'pixel'
    LAYER = 'layer'
    SPADE = 'spade'


class GenDisNormTypeEnum(Enum):
    SPECTRAL = 'spectral'
    WEIGHT = 'weight'


class ReluTypeEnum(Enum):
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'
    PRELU = 'prelu'
    SELU = 'selu'


class ScaleTypeEnum(Enum):
    UP = 'up'
    DOWN = 'down'
    DOWN_AVG = 'down_avg'


class InitWeightTypeEnum(Enum):
    NORMAL = 'normal'
    XAVIER = 'xavier'
    KAIMING = 'kaiming'
    ORTHOGONAL = 'orthogonal'


class GANModeEnum(Enum):
    LSGAN = 'lsgan'
    VANILLA = 'vanilla'
    HINGE = 'hinge'
    WGANGP = 'wgangp'


class PerceptualLossBaseModelEnum(Enum):
    VGG_19 = 'vgg_19'
    RESNET_50 = 'resnet_50'


class ModelNameEnum(Enum):
    GENERATOR = 'gen'
    DISCRIMINATOR = 'disc'
    FPN = 'fpn'


class LossNameEnum(Enum):
    PIX = 'pix'
    PERCEPTUAL = 'pcp'
    GENERATOR = 'gen'
    FEATURE_MATCHING = 'fm'
    DISCRIMINATOR = 'disc'
    SEMANTIC_STYLE = 'ss'
    FPN = 'fpn'


class LearningRatePolicyEnum(Enum):
    LINEAR = 'linear'
    STEP = 'step'
    PLATEAU = 'plateau'
    COSINE = 'cosine'


class MainModelNameEnum(Enum):
    PSFRGAN = 'psfrgan'
    FPN = 'fpn'


class PhaseEnum(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
