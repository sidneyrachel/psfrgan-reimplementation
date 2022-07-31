import torch
import os
from tqdm import tqdm
from PIL import Image

from util.config import Config
from util.common import make_directories, batch_tensor_to_image, colorize_mask
from util.create import create_dataset, create_model


if __name__ == '__main__':
    config = Config(
        filename='./config/psfrgan/test.json'
    )

    dataset = create_dataset(config)
    model = create_model(config)
    model.load_pretrain_models()

    make_directories(config.destination_directory)

    fpn_model = model.fpn_model
    gen_model = model.gen_model
    model.eval()

    low_res_directory = os.path.join(config.destination_directory, 'lr')
    make_directories(low_res_directory)
    high_res_directory = os.path.join(config.destination_directory, 'sr')
    make_directories(high_res_directory)
    mask_directory = os.path.join(config.destination_directory, 'mask')
    make_directories(mask_directory)

    for current_iteration, data in tqdm(enumerate(dataset), total=len(dataset) // config.batch_size):
        inp = data['lr']

        with torch.no_grad():
            low_res_mask, _ = fpn_model(inp)
            one_hot_low_res_mask = (low_res_mask == low_res_mask.max(dim=1, keepdim=True)[0]).float()
            super_res_image = gen_model(inp, one_hot_low_res_mask)

        image_paths = data['lr_path']  # Image paths in one batch
        low_res_images = batch_tensor_to_image(inp)
        super_res_images = batch_tensor_to_image(super_res_image)
        low_res_masks = colorize_mask(one_hot_low_res_mask)

        for i in tqdm(range(len(image_paths))):
            save_path = os.path.join(low_res_directory, os.path.basename(image_paths[i]))
            save_image = Image.fromarray(low_res_images[i])
            save_image.save(save_path)

            save_path = os.path.join(high_res_directory, os.path.basename(image_paths[i]))
            save_image = Image.fromarray(super_res_images[i])
            save_image.save(save_path)

            save_path = os.path.join(mask_directory, os.path.basename(image_paths[i]))
            save_image = Image.fromarray(low_res_masks[i])
            save_image.save(save_path)
