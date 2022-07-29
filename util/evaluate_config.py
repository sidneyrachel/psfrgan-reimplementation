import torch
import json

from util.common import get_gpu_memory_map


class EvaluateConfig:
    def __init__(self, filename):
        config_file = open(filename)
        self.config = json.load(config_file)

        if self.config['num_gpu'] > 0:
            self.config['gpu_ids'] = get_gpu_memory_map()[1][:self.config['num_gpu']]

            if not isinstance(self.config['gpu_ids'], list):
                self.config['gpu_ids'] = [self.config['gpu_ids']]

            torch.cuda.set_device(self.config['gpu_ids'][0])

            self.config['device'] = torch.device(f'cuda:{self.config["gpu_ids"][0 % self.config["num_gpu"]]}')
        else:
            self.config['gpu_ids'] = []
            self.config['device'] = torch.device('cpu')

    @property
    def low_res_size(self):
        return self.config['low_res_size']

    @property
    def device(self):
        return self.config['device']

    @property
    def gpu_ids(self):
        return self.config['gpu_ids']

    @property
    def num_gpu(self):
        return self.config['num_gpu']

    @property
    def high_res_size(self):
        return self.config['high_res_size']

    @property
    def max_data_count(self):
        return self.config['max_data_count']

    @property
    def ground_truth_dataset_path(self):
        return self.config['ground_truth_dataset_path']

    @property
    def predicted_dataset_path(self):
        return self.config['predicted_dataset_path']

    @property
    def dataset_name(self):
        return self.config['dataset_name']

    @property
    def batch_size(self):
        return self.config['batch_size']

    @property
    def is_shuffled(self):
        return self.config['is_shuffled']

    @property
    def num_thread(self):
        return self.config['num_thread']

    @property
    def is_train(self):
        return self.config['is_train']

    @property
    def ext(self):
        return self.config['ext']
