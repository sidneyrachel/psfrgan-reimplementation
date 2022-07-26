from variable.model import MainModelNameEnum
from dataset.custom_dataloader import CustomDataLoader
from model.psfrgan_model import PSFRGANModel
from model.fpn_model import FPNModel


def create_dataset(config):
    data_loader = CustomDataLoader(config)
    dataset = data_loader.load_data()

    return dataset


def create_model(config):
    if config.main_model_name == MainModelNameEnum.PSFRGAN:
        return PSFRGANModel(config)
    elif config.main_model_name == MainModelNameEnum.FPN:
        return FPNModel(config)
    else:
        raise Exception(f'Main model name is not supported. Main model name: {config.main_model_name}.')
