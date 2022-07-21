import torch
from torch.utils.data import DataLoader

from variable.dataset import DatasetNameEnum
from ffhq_dataset import FFHQDataset


class CustomDataLoader:
    def __init__(self, config):
        self.dataset_name = config.dataset_name

        if self.dataset_name == DatasetNameEnum.FFHQ:
            self.dataset = FFHQDataset(config)

        drop_last = config.is_train

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=config.is_shuffled,
            num_worker=int(config.num_thread),
            drop_last=drop_last
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self.dataloader
