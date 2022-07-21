import os
import torch

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from PIL import Image

from dataset.util import make_dataset, apply_random_gray, run_image_augmentation, convert_to_one_hot


class FFHQDataset(Dataset):
    def __init__(self, config):
        self.low_res_size = config.low_res_size
        self.high_res_size = config.high_res_size
        self.is_shuffle = config.is_train
        self.max_data_count = config.max_data_count
        self.data_base_path = config.data_base_path

        self.image_paths, filename_set = make_dataset(
            path=os.path.join(self.data_base_path, 'imgs1024'),
            max_data_count=self.max_data_count
        )

        self.mask_paths, _ = make_dataset(
            path=os.path.join(self.data_base_path, 'masks512'),
            filename_set=filename_set
        )

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        mask_path = self.mask_paths[item]

        high_res_image = Image.open(image_path).convert('RGB')
        high_res_image = high_res_image.resize((self.high_res_size, self.high_res_size))
        high_res_image = apply_random_gray(high_res_image, prob=0.3)
        high_res_tensor = self.to_tensor(high_res_image)

        low_res_image = run_image_augmentation(image=high_res_image, high_res_size=self.high_res_size)
        low_res_tensor = self.to_tensor(low_res_image)

        mask_image = Image.open(mask_path).convert('RGB')
        mask_image = mask_image.resize((self.high_res_size, self.high_res_size))
        mask_label = convert_to_one_hot(mask_image)
        mask_label = torch.tensor(mask_label).float()

        return {
            'hr': high_res_tensor,
            'lr': low_res_tensor,
            'hr_path': image_path,
            'mask': mask_label
        }
