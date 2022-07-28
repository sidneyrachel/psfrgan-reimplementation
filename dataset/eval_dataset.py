import os

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

from dataset.util import make_dataset


class EvalDataset(Dataset):
    def __init__(self, config):
        self.ground_truth_dataset_path = config.ground_truth_dataset_path
        self.predicted_dataset_path = config.predicted_dataset_path

        self.hr_paths, _ = make_dataset(
            path=self.ground_truth_dataset_path,
            max_data_count=config.max_data_count,
            ext=config.ext
        )

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        hr_path = self.hr_paths[item]
        hr_image = Image.open(hr_path).convert('RGB')
        hr_tensor = self.to_tensor(hr_image)

        sr_path = os.path.join(self.predicted_dataset_path, os.path.basename(hr_path))
        sr_image = Image.open(sr_path).convert('RGB')
        sr_tensor = self.to_tensor(sr_image)

        return {
            'hr': hr_tensor,
            'sr': sr_tensor
        }

    def __len__(self):
        return len(self.hr_paths)
