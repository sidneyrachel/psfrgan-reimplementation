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

    @property
    def dataset_name(self):
        return self.config['dataset_name']

    @property
    def is_shuffled(self):
        return self.config['is_shuffled']

    @property
    def num_thread(self):
        return self.config['num_thread']

    @property
    def p_norm_type(self):
        return self.config['p_norm_type']

    @property
    def num_gpu(self):
        return self.config['num_gpu']

    @property
    def fpn_pretrained_weight_file(self):
        return self.config['fpn_pretrained_weight_file']

    @property
    def psfr_pretrained_weight_file(self):
        return self.config['psfr_pretrained_weight_file']

    @property
    def generator_out_size(self):
        return self.config['generator_out_size']

    @property
    def generator_norm(self):
        return self.config['generator_norm']

    @property
    def num_discriminator_filter(self):
        return self.config['num_discriminator_filter']

    @property
    def num_discriminator_layer(self):
        return self.config['num_discriminator_layer']

    @property
    def discriminator_norm(self):
        return self.config['discriminator_norm']

    @property
    def num_discriminator(self):
        return self.config['num_discriminator']

    @property
    def discriminator_in_channel(self):
        return self.config['discriminator_in_channel']

    @property
    def vgg_pretrained_weight_file(self):
        return self.config['vgg_pretrained_weight_file']

    @property
    def gan_mode(self):
        return self.config['gan_mode']

    @property
    def generator_learning_rate(self):
        return self.config['generator_learning_rate']

    @property
    def discriminator_learning_rate(self):
        return self.config['discriminator_learning_rate']

    @property
    def beta1(self):
        return self.config['beta1']

    @property
    def debug(self):
        return self.config['debug']

    @property
    def pix_lambda(self):
        return self.config['pix_lambda']

    @property
    def ss_lambda(self):
        return self.config['ss_lambda']

    @property
    def perceptual_lambda(self):
        return self.config['perceptual_lambda']

    @property
    def fm_lambda(self):
        return self.config['fm_lambda']

    @property
    def gen_lambda(self):
        return self.config['gen_lambda']

    @property
    def fpn_lambda(self):
        return self.config['fpn_lambda']

    @property
    def fpn_pix_lambda(self):
        return self.config['fpn_pix_lambda']

    @property
    def fpn_learning_rate(self):
        return self.config['fpn_learning_rate']
