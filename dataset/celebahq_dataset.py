import os

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from PIL import Image

from dataset.util import make_dataset, apply_random_gray, run_image_augmentation


class CelebAHQDataset(Dataset):
    def __init__(self, config):
        self.low_res_size = config.low_res_size
        self.high_res_size = config.high_res_size
        self.max_data_count = config.max_data_count
        self.dataset_base_path = config.dataset_base_path

        self.image_paths, filename_set = make_dataset(
            path=os.path.join(self.dataset_base_path, 'imgs1024'),
            max_data_count=self.max_data_count,
            ext='.jpg',
            is_random=True
        )

        print(f'number of images: {len(self.image_paths)}')

        self.image_paths = sorted(self.image_paths)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]

        high_res_image = Image.open(image_path).convert('RGB')
        high_res_image = high_res_image.resize((self.high_res_size, self.high_res_size))
        # TODO: Check if this is true?
        # high_res_image = apply_random_gray(high_res_image, prob=0.3)

        low_res_image = run_image_augmentation(image=high_res_image, high_res_size=self.high_res_size)

        high_res_tensor = self.to_tensor(high_res_image)
        low_res_tensor = self.to_tensor(low_res_image)

        return {
            'hr': high_res_tensor,
            'lr': low_res_tensor,
            'hr_path': image_path
        }
