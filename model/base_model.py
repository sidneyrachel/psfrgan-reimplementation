import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from variable.model import LearningRatePolicyEnum
from model.util import get_scheduler


class BaseModel(ABC):
    def __init__(self, config):
        """Initialize the BaseModel class.

        Parameters:
            config -- Stores all the experiment variables.

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, config)>.
        Then, you need to define four lists:
            -- self.loss_names (str list): Specify the training losses that you want to plot and save.
            -- self.model_names (str list): Define networks used in our training.
            -- self.visual_names (str list): Specify the images that you want to display and save.
            -- self.optimizers (optimizer list): Define and initialize optimizers.
        """
        self.config = config
        self.gpu_ids = config.gpu_ids
        self.is_train = config.is_train
        self.checkpoint_directory = os.path.join(config.checkpoint_directory_path, config.experiment_name)
        self.device = torch.device(f'cuda:{self.gpu_ids[0]}') if self.gpu_ids \
            else torch.device('cpu')

        self.loss_names = []
        self.model_names = []
        self.load_model_names = []
        # self.visual_names = []
        self.optimizers = []
        self.image_paths = []

        self.metric = 0  # Used for learning rate policy 'plateau'.

    @abstractmethod
    def set_input(self, inp, current_iter):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self, config):
        if self.is_train:
            self.schedulers = [get_scheduler(optimizer, config) for optimizer in self.optimizers]

        if not self.is_train or config.continue_train:
            if config.load_iter > 0:
                prefix = f'iter_{config.load_iter}'
            else:
                prefix = f'epoch_{config.load_epoch}'

            self.load_networks(prefix)

        self.print_networks()

    def eval(self):
        for model_name in self.model_names:
            if isinstance(model_name, str):
                network = getattr(self, f'{model_name}_model')
                network.eval()

    def test(self):
        with torch.no_grad():
            self.forward()
            self.generate_visual_images()

    def generate_visual_images(self):
        pass

    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.config.learning_rate_policy == LearningRatePolicyEnum.PLATEAU.value:
                scheduler.step(self.metric)
            else:
                scheduler.step()

        learning_rate = self.optimizers[0].param_groups[0]['lr']

        print(f'Current learning rate: {learning_rate}.')

    def get_learning_rate(self):
        learning_rate_map = {}

        for idx, optimizer in enumerate(self.optimizers):
            learning_rate_map[f'LR{idx}'] = optimizer.param_groups[0]['lr']

        return learning_rate_map

    def get_current_visual_images(self, size):
        pass

    def get_current_losses(self):
        loss_map = OrderedDict()

        for loss_name in self.loss_names:
            if isinstance(loss_name, str):
                loss_map[f'{loss_name}_loss'] = float(getattr(self, f'{loss_name}_loss'))

        return loss_map

    def save_networks(self, prefix, info=None):
        for model_name in self.model_names:
            if isinstance(model_name, str):
                save_filename = f'{prefix}_net_{model_name}.pth'
                model_path = os.path.join(self.checkpoint_directory, save_filename)
                network = getattr(self, f'{model_name}_model')

                if isinstance(network, torch.nn.DataParallel) or isinstance(network, torch.nn.parallel.DistributedDataParallel):
                    torch.save(network.module.cpu().state_dict(), model_path)
                    print(f'Model saved in {model_path}.')
                    network.cuda(self.gpu_ids[0])
                else:
                    torch.save(network.cpu().state_dict(), model_path)

        if info is not None:
            torch.save(info, os.path.join(self.checkpoint_directory, f'{prefix}.info'))

    def load_networks(self, prefix):
        for model_name in self.load_model_names:
            if isinstance(model_name, str):
                load_filename = f'{prefix}_net_{model_name}.pth'
                model_path = os.path.join(self.checkpoint_directory, load_filename)
                network = getattr(self, f'{model_name}_model')

                if isinstance(network, torch.nn.DataParallel) or isinstance(network, torch.nn.parallel.DistributedDataParallel):
                    network = network.module

                print(f'Load the model from {model_path}.')

                state_dict = torch.load(model_path, map_location=str(self.device))

                if self.config.is_strict_load:
                    network.load_state_dict(state_dict)
                else:  # Load partial weights
                    model_dict = network.state_dict()
                    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    network.load_state_dict(model_dict, strict=False)

        info_path = os.path.join(self.checkpoint_directory, f'{prefix}.info')

        if os.path.exists(info_path):
            info_dict = torch.load(info_path)
            for k, v in info_dict.items():
                setattr(self.config, k, v)

    def print_networks(self):
        for model_name in self.model_names:
            if isinstance(model_name, str):
                net = getattr(self, f'{model_name}_model')
                num_params = 0

                for param in net.parameters():
                    num_params += param.numel()

                print(f'[{model_name}] Total number of parameters: {num_params / 1e6} millions.')

    @staticmethod
    def set_requires_grad(networks, requires_grad=False):
        if not isinstance(networks, list):
            networks = [networks]

        for network in networks:
            for param in network.parameters():
                param.requires_grad = requires_grad
