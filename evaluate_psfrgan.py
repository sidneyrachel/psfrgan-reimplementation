from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, \
    MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from cleanfid import fid

from util.evaluate_config import EvaluateConfig
from util.create import create_dataset


if __name__ == '__main__':
    config = EvaluateConfig(
        filename='./config/evaluate.json'
    )

    dataset = create_dataset(config)

    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarityIndexMeasure()
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

    fid_score = fid.compute_fid(config.predicted_dataset_path, config.ground_truth_dataset_path)
    print('FID:', fid_score)

    for data in dataset:
        print('PSNR:', psnr(data['sr'], data['hr']))
        print('SSIM:', ssim(data['sr'], data['hr']))
        print('MS-SSIM:', ms_ssim(data['sr'], data['hr']))
        print('LPIPS:', lpips(data['sr'], data['hr']))
