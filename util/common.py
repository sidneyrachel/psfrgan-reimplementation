import numpy as np
import cv2
import os


from variable.mask import MASK_COLORMAP


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
