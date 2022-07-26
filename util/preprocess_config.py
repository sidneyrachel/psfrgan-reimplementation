import json


class PreprocessConfig:
    def __init__(self, filename):
        config_file = open(filename)
        self.config = json.load(config_file)

    @property
    def source_directory(self):
        return self.config['source_directory']

    @property
    def destination_directory(self):
        return self.config['destination_directory']

    @property
    def is_cnn_detector(self):
        return self.config['is_cnn_detector']

    @property
    def cnn_face_detector_file(self):
        return self.config['cnn_face_detector_file']

    @property
    def shape_detector_file(self):
        return self.config['shape_detector_file']

    @property
    def template_file(self):
        return self.config['template_file']

    @property
    def template_scale(self):
        return self.config['template_scale']
