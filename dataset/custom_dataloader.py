from torch.utils.data import DataLoader

from variable.dataset import DatasetNameEnum
from dataset.ffhq_dataset import FFHQDataset
from dataset.test_dataset import TestDataset


class CustomDataLoader:
    def __init__(self, config):
        self.dataset_name = config.dataset_name

        if self.dataset_name == DatasetNameEnum.FFHQ.value:
            self.dataset = FFHQDataset(config)
        elif self.dataset_name == DatasetNameEnum.TEST.value:
            self.dataset = TestDataset(config)
        else:
            raise Exception(f'Dataset name is not supported. Dataset name: {self.dataset_name}.')

        drop_last = config.is_train

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=config.is_shuffled,
            num_workers=int(config.num_thread),
            drop_last=drop_last
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
