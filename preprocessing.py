from util.preprocess_config import PreprocessConfig
from model.preprocessing import Preprocessing

if __name__ == '__main__':
    config = PreprocessConfig(
        filename='config/preprocessing.json'
    )

    preprocessing = Preprocessing(config)
    preprocessing.align_and_crop_faces()
