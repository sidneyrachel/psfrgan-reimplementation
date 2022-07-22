from enum import Enum


class NormTypeEnum(Enum):
    BN = 'bn',
    IN = 'in',
    GN = 'gn',
    PIXEL = 'pixel',
    LAYER = 'layer',
    SPADE = 'spade'


class GenDisNormTypeEnum(Enum):
    SPECTRAL = 'spectral',
    WEIGHT = 'weight'


class ReluTypeEnum(Enum):
    RELU = 'relu',
    LEAKY_RELU = 'leaky_relu',
    PRELU = 'prelu',
    SELU = 'selu'


class ScaleTypeEnum(Enum):
    UP = 'up',
    DOWN = 'down',
    DOWN_AVG = 'down_avg'


class InitWeightTypeEnum(Enum):
    NORMAL = 'normal',
    XAVIER = 'xavier',
    KAIMING = 'kaiming',
    ORTHOGONAL = 'orthogonal'


class GANModeEnum(Enum):
    LSGAN = 'lsgan',
    VANILLA = 'vanilla',
    HINGE = 'hinge',
    WGANGP = 'wgangp'


class PerceptualLossBaseModelEnum(Enum):
    VGG_19 = 'vgg_19',
    RESNET_50 = 'resnet_50'


class ModelNameEnum(Enum):
    GENERATOR = 'generator',
    DISCRIMINATOR = 'discriminator',
    FPN = 'fpn'


class LossNameEnum(Enum):
    PIX = 'pix',
    PERCEPTUAL = 'perceptual',
    GENERATOR = 'generator',
    FEATURE_MATCHING = 'feature_matching',
    DISCRIMINATOR = 'discriminator',
    SEMANTIC_STYLE = 'semantic_style',
    FPN = 'fpn'


class VisualNameEnum(Enum):
    LOW_RES_IMAGE = 'low_res_image',
    HIGH_RES_IMAGE = 'high_res_image',
    SUPER_RES_IMAGE = 'super_res_image'
    LOW_RES_MASK = 'low_res_mask',
    HIGH_RES_MASK = 'high_res_mask',
    PREDICTED_MASK = 'predicted_mask',
    GROUND_TRUTH_MASK = 'ground_truth_mask'
