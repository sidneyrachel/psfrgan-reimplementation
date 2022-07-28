from tqdm import tqdm
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

    psnr_total = 0
    ssim_total = 0
    ms_ssim_total = 0
    lpips_total = 0

    num_batch = len(dataset) / config.batch_size

    for data in tqdm(dataset):
        psnr_total += psnr(data['sr'], data['hr'])
        ssim_total += ssim(data['sr'], data['hr'])
        ms_ssim_total += ms_ssim(data['sr'], data['hr'])
        lpips_total += lpips(data['sr'], data['hr'])

    print('PSNR:', psnr_total / num_batch)
    print('SSIM:', ssim_total / num_batch)
    print('MS-SSIM:', ms_ssim_total / num_batch)
    print('LPIPS:', lpips_total / num_batch)
    print('FID:', fid_score)
