import json


class TestGenerationConfig:
    def __init__(self, filename):
        config_file = open(filename)
        self.config = json.load(config_file)

    @property
    def low_res_size(self):
        return self.config['low_res_size']

    @property
    def high_res_size(self):
        return self.config['high_res_size']

    @property
    def max_data_count(self):
        return self.config['max_data_count']

    @property
    def dataset_base_path(self):
        return self.config['dataset_base_path']

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
