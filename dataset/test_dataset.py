from PIL import Image
from torch.utils.data import Dataset

from dataset.util import make_dataset
from model.util import compose_transform


class TestDataset(Dataset):
    def __init__(self, config):
        self.image_paths, _ = make_dataset(
            path=config.dataset_base_path,
            max_data_count=config.max_data_count
        )

        self.image_paths = sorted(self.image_paths)

        in_channel = config.test_in_channel
        self.transform = compose_transform(config, grayscale=(in_channel == 1))

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512), Image.BICUBIC)

        transformed_image = self.transform(image)

        return {
            'lr': transformed_image,
            'lr_path': image_path
        }

    def __len__(self):
        return len(self.image_paths)
