import torch
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import multiscale_structural_similarity_index_measure, \
    structural_similarity_index_measure

from util.evaluate_config import EvaluateConfig
from util.create import create_dataset


if __name__ == '__main__':
    config = EvaluateConfig(
        filename='./config/evaluate.json'
    )

    dataset = create_dataset(config)

    dataset_size = len(dataset)
    data_range = 2

    psnr = PeakSignalNoiseRatio(
        compute_on_cpu=True,
        data_range=data_range
    )
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='vgg',
        compute_on_cpu=True,
        data_range=data_range
    )

    if len(config.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        psnr = psnr.to(config.device)
        lpips = lpips.to(config.device)

    ssim_total = 0
    ms_ssim_total = 0

    num_batch = dataset_size // config.batch_size

    print(f'Total batch: {num_batch}.')

    for i, data in tqdm(enumerate(dataset)):
        print(f'[{i}/{num_batch}] Calculate metrics.')
        data['sr'] = data['sr'].to(config.device)
        data['hr'] = data['hr'].to(config.device)

        psnr(data['sr'], data['hr'])
        lpips(data['sr'], data['hr'])

        ssim_total += (
                structural_similarity_index_measure(
                    data['sr'],
                    data['hr'],
                    data_range=data_range
                ) * config.batch_size
        )
        ms_ssim_total += (
                multiscale_structural_similarity_index_measure(
                    data['sr'],
                    data['hr'],
                    data_range=data_range
                ) * config.batch_size
        )

    print('PSNR:', psnr.compute())
    print('LPIPS:', lpips.compute())
    print('SSIM:', ssim_total / dataset_size)
    print('MS-SSIM:', ms_ssim_total / dataset_size)
