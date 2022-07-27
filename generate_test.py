from tqdm import tqdm

from util.test_generation_config import TestGenerationConfig
from util.create import create_dataset


if __name__ == '__main__':
    config = TestGenerationConfig(
        filename='./config/test_generation.json'
    )

    dataset = create_dataset(config)

    for i, data in tqdm(enumerate(dataset)):
        print(f'[{i}/{len(dataset)}]Generate lr-hr images.')
