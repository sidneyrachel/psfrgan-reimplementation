# Reimplementation of Progressive Semantic-Aware Style Transformation for Blind Face Restoration
Full explanation of this project can be found at <insert-blog>.

## Prerequisite
1. Python 3.7
2. CUDA 10.1
3. torch==1.5.1 
4. torchvision==0.6.1 
5. tensorflow>=1.15.4 
6. opencv-python 
7. dlib 
8. scipy==1.4.1 
9. tqdm 
10. imgaug
11. torchmetrics
12. pytorch-fid


## Training
The script to train the model using FFHQ dataset is available at `train.py`. The config for running this script can be found at `config/base.json` and `config/psfrgan/train.json`.

## Downsample Test Dataset
The script to generate low-resolution (LR) face images and ground truth high-resolution (HR) face images from 1024x1024 CelebAHQ dataset for testing purpose is available at `downsample_psfrgan.py`. The config for running this script can be found at `config/downsample.json`.

## Testing
The script to generate the predicted super-resolution (SR) face images is available at `test_psfrgan.py`. The config for running this script can be found at `config/base.json` and `config/psfrgan/test.json`.

## Evaluation
The script to evaluate the predicted SR face images against ground truth HR face images is available at `evaluate_psfrgan.py`. The config for running this script can be found at `config/evaluate.json`. The script will produce PSNR, LPIPS, and SSIM score. To retrieve FID score, simply run `python -m pytorch_fid <path/to/sr-folder> <path/to/hr-folder>`.

## Test your Own Image
The script to prepare your own test images from real LR face images is available at `preprocess_psfrgan.py`. This script will crop and align the real faces so that it is ready to be used to evaluate the model. The config for running this script is available at `config/preprocess.json`.
