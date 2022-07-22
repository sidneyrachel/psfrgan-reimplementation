import os
import sys

import numpy as np

import imgaug
from imgaug import augmenters

from variable.mask import MASK_COLORMAP
from dataset.custom_dataloader import CustomDataLoader


def make_dataset(
        path,
        max_data_count=None,
        filename_set=None
):
    paths = []
    filenames = []

    for root, _, sub_filenames in sorted(os.walk(path)):
        for filename in sub_filenames:
            if filename_set and (filename not in filename_set):
                continue

            filenames.append(filename)

            path = os.path.join(root, filename)
            paths.append(path)

    count = min(max_data_count, len(paths)) if max_data_count else len(paths)

    truncated_paths, truncated_filename_set = paths[:count], set(filenames[:count])

    if len(truncated_paths) != len(truncated_filename_set):
        raise Exception(f'Paths and filenames mismatch.'
                        f'Paths length: {len(truncated_paths)}.'
                        f'Filenames length: {len(truncated_filename_set)}.')

    return truncated_paths, truncated_filename_set


def apply_random_gray(image, prob=0.5):
    images = np.array(image)
    images = images[np.newaxis, :, :, :]
    augmentation = augmenters.Sometimes(prob, augmenters.Grayscale(alpha=1.0))
    augmented_images = augmentation(images=images)

    return augmented_images[0]


def run_image_augmentation(
        image,
        high_res_size
):
    images = np.array(image)
    images = images[np.newaxis, :, :, :]

    scale_size = np.random.randint(32, 256)

    augmentation_sequence = augmenters.Sequential([
        augmenters.Sometimes(0.5, augmenters.OneOf([
            augmenters.GaussianBlur((3, 15)),
            augmenters.AverageBlur(k=(3, 15)),
            augmenters.MedianBlur(k=(3, 15)),
            augmenters.MotionBlur((5, 25))
        ])),
        augmenters.Resize(scale_size, interpolation=imgaug.ALL),
        augmenters.Sometimes(0.2, augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 25.5), per_channel=0.5)),
        augmenters.Sometimes(0.7, augmenters.JpegCompression(compression=(10, 65))),
        augmenters.Resize(high_res_size)
    ])

    augmented_images = augmentation_sequence(images=images)

    return augmented_images[0]


def convert_to_one_hot(image):
    num_label = len(MASK_COLORMAP)
    image = np.array(image, dtype=np.unit8)
    height, width = image.shape[:2]
    one_hot_label = np.zeros((num_label, height, width))
    colormap = np.array(MASK_COLORMAP).reshape(num_label, 1, 1, 3)
    colormap = np.tile(colormap, (1, height, width, 1))  # output dimension: (num_label, height, width, 3)

    for idx, color in enumerate(MASK_COLORMAP):
        label = (colormap[idx] == image)
        one_hot_label[idx] = label[..., 0] * label[..., 1] * label[..., 2]

    return one_hot_label


def create_dataset(config):
    data_loader = CustomDataLoader(config)
    dataset = data_loader.load_data()

    return dataset
