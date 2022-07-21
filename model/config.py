import json


class Config:
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
    def is_train(self):
        return self.config['is_train']

    @property
    def max_data_count(self):
        return self.config['max_data_count']

    @property
    def data_base_path(self):
        return self.config['data_base_path']