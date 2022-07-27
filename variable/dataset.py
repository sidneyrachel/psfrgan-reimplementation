from enum import Enum


class DatasetNameEnum(Enum):
    FFHQ = 'ffhq'
    TEST = 'test'
    CELEB_A_HQ = 'celeb_a_hq'


class PreprocessActionEnum(Enum):
    RESIZE_AND_CROP = 'resize_and_crop'
    CROP = 'crop'
    SCALE_WIDTH = 'scale_width'
    SCALE_WIDTH_AND_CROP = 'scale_width_and_crop'
