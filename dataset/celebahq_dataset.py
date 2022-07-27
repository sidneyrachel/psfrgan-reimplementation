import os
import torch

from torch.utils.data import Dataset

from PIL import Image

from dataset.util import make_dataset, apply_random_gray, run_image_augmentation
from util.common import make_directories


class CelebAHQDataset(Dataset):
    def __init__(self, config):
        self.low_res_size = config.low_res_size
        self.high_res_size = config.high_res_size
        self.max_data_count = config.max_data_count
        self.dataset_base_path = config.dataset_base_path

        self.image_paths, filename_set = make_dataset(
            path=os.path.join(self.dataset_base_path, 'imgs1024'),
            max_data_count=self.max_data_count,
            ext='.jpg'
        )

        self.lr_dir = os.path.join(self.dataset_base_path, 'lr')
        self.hr_dir = os.path.join(self.dataset_base_path, 'hr')

        make_directories([
            self.lr_dir,
            self.hr_dir
        ])

        print(f'number of images: {len(self.image_paths)}')

        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]

        high_res_image = Image.open(image_path).convert('RGB')
        high_res_image = high_res_image.resize((self.high_res_size, self.high_res_size))
        high_res_image = apply_random_gray(high_res_image, prob=0.3)

        low_res_image = run_image_augmentation(image=high_res_image, high_res_size=self.high_res_size)

        saved_hr_image = Image.fromarray(high_res_image)
        saved_hr_image.save(os.path.join(self.hr_dir, os.path.basename(image_path)))

        saved_lr_image = Image.fromarray(low_res_image)
        saved_lr_image.save(os.path.join(self.lr_dir, os.path.basename(image_path)))

        return {
            'hr': high_res_image,
            'lr': low_res_image
        }
