import numpy as np
import cv2
import os

from variable.model import MainModelNameEnum
from variable.mask import MASK_COLORMAP
from dataset.custom_dataloader import CustomDataLoader
from model.psfrgan_model import PSFRGANModel
from model.fpn_model import FPNModel


def tensor_to_numpy(tensor):
    return tensor.data.cpu().numpy()


# Array shape: (B, C, H, W)
def batch_numpy_to_image(array, size=None):
    if isinstance(size, int):
        size = (size, size)

    out_images = []
    array = np.clip((array + 1)/2 * 255, 0, 255)
    array = np.transpose(array, (0, 2, 3, 1))

    for i in range(array.shape[0]):
        if size is not None:
            resized_array = cv2.resize(array[i], size)
        else:
            resized_array = array[i]

        out_images.append(resized_array)

    return np.array(out_images).astype(np.uint8)


# Input shape: (B, C, H, W)
# Return image: RGB (0, 255)
def batch_tensor_to_image(tensor, size=None):
    array = tensor_to_numpy(tensor)
    out_images = batch_numpy_to_image(array, size)

    return out_images


def colorize_mask(tensor, size=None):
    if len(tensor.shape) < 4:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[1] > 1:
        tensor = tensor.argmax(dim=1)

    tensor = tensor.squeeze(1).data.cpu().numpy()

    color_masks = []

    for t in tensor:
        color_mask = np.zeros(tensor.shape[1:] + (3,))

        for idx, color in enumerate(MASK_COLORMAP):
            color_mask[t == idx] = color

        if size is not None:
            color_mask = cv2.resize(color_mask, (size, size))

        color_masks.append(color_mask.astype(np.uint8))

    return color_masks


def make_directories(directories):
    if isinstance(directories, list) and not isinstance(directories, str):
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
    else:
        if not os.path.exists(directories):
            os.makedirs(directories)


def create_dataset(config):
    data_loader = CustomDataLoader(config)
    dataset = data_loader.load_data()

    return dataset


def create_model(config):
    if config.main_model_name == MainModelNameEnum.PSFRGAN:
        return PSFRGANModel(config)
    elif config.main_model_name == MainModelNameEnum.FPN:
        return FPNModel(config)
    else:
        raise Exception(f'Main model name is not supported. Main model name: {config.main_model_name}.')
