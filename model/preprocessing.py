import dlib
import os
import cv2
import numpy as np
from util.common import make_directories
from skimage import transform


class Preprocessing:
    def __init__(self, config):
        self.src_dir = config.source_directory
        self.dest_dir = config.destination_directory
        make_directories(self.dest_dir)

        if config.is_cnn_detector:
            self.face_detector = dlib.cnn_face_detection_model_v1(config.cnn_face_detector_file)
        else:
            self.face_detector = dlib.get_frontal_face_detector()

        self.shape_predictor = dlib.shape_predictor(config.shape_detector_file)
        self.reference_alignment = np.load(config.template_file) / config.template_scale

    def get_points(self, image, size_threshold=999):
        results = self.face_detector(image, 1)

        points = []

        for bbox in results:
            if isinstance(self.face_detector, dlib.cnn_face_detection_model_v1):
                rectangle = bbox.rect
            else:
                rectangle = bbox

            if rectangle.width() > size_threshold or rectangle.height() > size_threshold:
                break

            shape = self.shape_predictor(image, rectangle)

            individual_points = []

            for i in range(5):
                individual_points.append([shape.part(i).x, shape.part(i).y])

            points.append(np.array(individual_points))

        if len(points) == 0:
            return None
        else:
            return points

    def align_and_crop_face(self, image, save_path, source_points):
        out_size = (512, 512)

        extensions = os.path.splitext(save_path)

        for idx, source_point in enumerate(source_points):
            transform_tool = transform.SimilarityTransform()
            transform_tool.estimate(source_point, self.reference_alignment)
            M = transform_tool.params[0:2, :]

            cropped_image = cv2.warpAffine(image, M, out_size)

            if len(source_points) > 1:
                save_path = f'{extensions[0]}_{idx}{extensions[1]}'

            print(f'Saving image to {save_path}.')
            dlib.save_image(cropped_image.astype(np.uint8), save_path)

    def align_and_crop_faces(self):
        for filename in os.listdir(self.src_dir):
            image_path = os.path.join(self.src_dir, filename)
            image = dlib.load_rgb_image(image_path)

            points = self.get_points(image)

            if points is not None:
                save_path = os.path.join(self.dest_dir, filename)
                self.align_and_crop_face(image, save_path, points)
            else:
                print(f'No face detected in {image_path}.')
