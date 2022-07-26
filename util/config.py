import json


class Config:
    def __init__(self, filename, base_filename='./config/base.json'):
        base_config_file = open(base_filename)
        base_config = json.load(base_config_file)
        config_file = open(filename)
        config = json.load(config_file)
        base_config.update(config)

        self.config = base_config

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
    def dataset_base_path(self):
        return self.config['dataset_base_path']

    # Enum: ffhq, test
    @property
    def dataset_name(self):
        return self.config['dataset_name']

    # Enum: psfrgan, fpn
    @property
    def main_model_name(self):
        return self.config['main_model_name']

    @property
    def is_shuffled(self):
        return self.config['is_shuffled']

    @property
    def num_thread(self):
        return self.config['num_thread']

    # Enum: train, test, val
    @property
    def phase(self):
        return self.config['phase']

    # Enum: in, bn, gn, pixel, spade, layer
    @property
    def fpn_norm_type(self):
        return self.config['fpn_norm_type']

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

    # Enum: in, bn, gn, pixel, spade, layer
    @property
    def generator_norm(self):
        return self.config['generator_norm']

    @property
    def num_discriminator_filter(self):
        return self.config['num_discriminator_filter']

    @property
    def num_discriminator_layer(self):
        return self.config['num_discriminator_layer']

    # Enum: in, bn, gn, pixel, spade, layer
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
    def test_in_channel(self):
        return self.config['test_in_channel']

    @property
    def vgg_pretrained_weight_file(self):
        return self.config['vgg_pretrained_weight_file']

    # Enum: lsgan, vanilla, hinge, wgangp
    @property
    def gan_mode(self):
        return self.config['gan_mode']

    @property
    def generator_learning_rate(self):
        return self.config['generator_learning_rate']

    @property
    def discriminator_learning_rate(self):
        return self.config['discriminator_learning_rate']

    # Momentum for adam optimizer
    @property
    def gen_disc_beta1(self):
        return self.config['gen_disc_beta1']

    @property
    def is_debug(self):
        return self.config['is_debug']

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

    @property
    def checkpoint_directory_path(self):
        return self.config['checkpoint_directory_path']

    @property
    def experiment_name(self):
        return self.config['experiment_name']

    @property
    def num_epoch(self):
        return self.config['num_epoch']

    @property
    def init_learning_rate_num_epoch(self):
        return self.config['init_learning_rate_num_epoch']

    @property
    def start_epoch(self):
        return self.config['start_epoch']

    @property
    def decay_next_num_epoch(self):
        return self.config['decay_next_num_epoch']

    @property
    def gamma_decay_iter(self):
        return self.config['gamma_decay_iter']

    # Enum: linear, step, plateau, cosine
    @property
    def learning_rate_policy(self):
        return self.config['learning_rate_policy']

    @property
    def load_iter(self):
        return self.config['load_iter']

    @property
    def load_epoch(self):
        return self.config['load_epoch']

    @property
    def is_strict_load(self):
        return self.config['is_strict_load']

    @property
    def log_directory(self):
        return self.config['log_directory']

    @property
    def log_archive_directory(self):
        return self.config['log_archive_directory']

    @property
    def continue_train(self):
        return self.config['continue_train']

    @property
    def batch_size(self):
        return self.config['batch_size']

    @property
    def resume_iter_on_top_resume_epoch(self):
        return self.config['resume_iter_on_top_resume_epoch']

    @property
    def resume_epoch(self):
        return self.config['resume_epoch']

    @property
    def print_iteration_frequency(self):
        return self.config['print_iteration_frequency']

    @property
    def visual_iteration_frequency(self):
        return self.config['visual_iteration_frequency']

    @property
    def save_iteration_frequency(self):
        return self.config['save_iteration_frequency']

    # Enum: resize_and_crop | crop | scale_width | scale_width_and_crop
    @property
    def preprocess(self):
        return self.config['preprocess']

    @property
    def scale_size(self):
        return self.config['scale_size']

    @property
    def crop_size(self):
        return self.config['crop_size']

    @property
    def is_flipped(self):
        return self.config['is_flipped']

    @property
    def destination_directory(self):
        return self.config['destination_directory']
