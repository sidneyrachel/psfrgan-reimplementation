import os
from tqdm import tqdm
from PIL import Image

from util.test_generation_config import TestGenerationConfig
from util.create import create_dataset
from util.common import batch_tensor_to_image, make_directories


if __name__ == '__main__':
    config = TestGenerationConfig(
        filename='./config/test_generation.json'
    )

    dataset = create_dataset(config)

    lr_dir = os.path.join(config.dataset_base_path, 'lr')
    hr_dir = os.path.join(config.dataset_base_path, 'hr')

    make_directories([
        lr_dir,
        hr_dir
    ])

    for i, data in tqdm(enumerate(dataset)):
        print(f'[{i}/{len(dataset)}] Generate lr-hr images.')

        high_res_images = batch_tensor_to_image(data['hr'])
        low_res_images = batch_tensor_to_image(data['lr'])

        image_paths = data['hr_path']

        for j in range(len(high_res_images)):
            saved_hr_image = Image.fromarray(high_res_images[j])
            saved_hr_image.save(os.path.join(hr_dir, os.path.basename(image_paths[j])))

            saved_lr_image = Image.fromarray(low_res_images[j])
            saved_lr_image.save(os.path.join(lr_dir, os.path.basename(image_paths[j])))
