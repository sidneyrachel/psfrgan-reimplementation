from torch.utils.data import DataLoader

from variable.dataset import DatasetNameEnum
from dataset.ffhq_dataset import FFHQDataset
from dataset.test_dataset import TestDataset
from dataset.celebahq_dataset import CelebAHQDataset
from dataset.eval_dataset import EvalDataset

class CustomDataLoader:
    def __init__(self, config):
        self.dataset_name = config.dataset_name

        if self.dataset_name == DatasetNameEnum.FFHQ.value:
            self.dataset = FFHQDataset(config)
        elif self.dataset_name == DatasetNameEnum.TEST.value:
            self.dataset = TestDataset(config)
        elif self.dataset_name == DatasetNameEnum.CELEB_A_HQ.value:
            self.dataset = CelebAHQDataset(config)
        elif self.dataset_name == DatasetNameEnum.EVAL.value:
            self.dataset = EvalDataset(config)
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
