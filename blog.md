# Reimplementation of Progressive Semantic-Aware Style Transformation for Blind Face Restoration

## Introduction

This paper which was written by Chaofeng Chen, Xiaoming Li, Lingbo Yang, Xianhui Lin, Lei Zhang, and Kwan-Yee K. Wong, was published at CVPR 2021. They propose a progressive semantic-aware style transformation network, called PSFR-GAN, for blind face restoration. They utilize multi-scale low-resolution (LR) face images and their semantic segmentation maps to recover high-resolution (HR) face images through semantic aware style transformation. Furthermore, they introduce semantic aware style loss which calculates the feature loss of each semantic region individually to improve the restoration of face textures.

The main contributions of the paper are:
1. They propose a novel multi-scale progressive framework for blind face restoration, i.e., PSFR-GAN which can make better use of multi-scale inputs pixel-wise and semantic-wise.
2. They introduce semantic aware style loss to improve the restoration of face textures in each semantic region and reduce the occurrence of artifacts.
3. Their model generalizes better to real LR face images than state-of-the-arts.
4. They introduce a pre-trained face parsing network (FPN) to generate the segmentation map of the LR face image so the model is able to generate HR face image given only LR face image. It makes the model more practical in real-world cases where the segmentation maps of LR face images barely exist.

The main architecture of PSFR-GAN is the generator network. We provide the architecture drawn in the paper here.

![Screen Shot 2022-08-01 at 15.15.27.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/14f79d58-5083-b1c6-7b20-a251b141e13d.png)

The generator network starts with a constant $F_0$ with size $Cx16x16$. The $F_0$ goes through several upsample residual blocks. Finally, the $F_6$ goes through the RGB convolution to predict the super-resolution (SR) face image. They formulate $F_i$ using the equation below, where $\Phi_{RES}$ is the residual convolution block, $\Phi_{UP}$ is the upsample residual block, and $\Phi_{ST}$ is the style transformation block.

```math
F_i = \Phi_{ST}(\Phi_{RES}(F_{i-1})), i=1 \\
F_i = \Phi_{ST}(\Phi_{UP}(F_{i-1})), 1 < i \le 6
```

Each of the style transformation block learns a parameter $\mathbf{y_i}=(\mathbf{y_{s,i}},\mathbf{y_{b,i}})$ from a scale of input pair which are LR face image and its segmentation map: $(I_L^i,I_P^i)$. The input pair is resized to the same size as $F_i$. They formulate the process in the style transformation block as follows, where $\Psi$ is a network composed of some convolutional layers which we will explain the detail in the implementation part. $\mu$ and $\sigma$ are the mean and standard deviation of the features. The style parameter $\mathbf{y_i}$ makes full use of color and texture information from $I_L$ as well as shape and semantic guidance from $I_P$ to enhance the input $F_i$.

```math
\mathbf{y_i}=\Psi(I_L^i,I_P^i) \\
F_i=\mathbf{y_{s,i}}\frac{\Phi_{UP}(F_{i-1})-\mu(\Phi_{UP}(F_{i-1}))}{\sigma(\Phi_{UP}(F_{i-1}))}+\mathbf{y_{b,i}}, 1 < i \le 6
```

## Implementation

We reimplement the PSFR-GAN and publish the code at [github.com/sidneyrachel/psfrgan-reimplementation](https://github.com/sidneyrachel/psfrgan-reimplementation). We implement the code using PyTorch library and refer to the original code available at [github.com/chaofengc/PSFRGAN](https://github.com/chaofengc/PSFRGAN).

### Core Architecture

The proposed architecture is divided into three main networks: Face Parsing Network (FPN), generator, and discriminator. In the following sections, we will explain the implementation of each network. Finally, we will explain how PSFR-GAN is constructed from these three networks.

#### Face Parsing Network (FPN)

FPN produces a semantic segmentation map of a face image that labels each region with different colors, e.g., left eye, right eye, nose, upper lip, bottom lip, neck, etc. PSFRGAN uses a pre-trained model of this network to produce a semantic segmentation map of a low-resolution (LR) face image at the training and inference step. Below is the detailed architecture of FPN. 

![fpn.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/d5d1eff3-3480-ea93-2fbc-51f62ac1ed23.png)

It consists of a type-1 convolutional layer (we will explain later about the type), four encoder residual blocks that downsample the image, ten body residual blocks, and four decoder residual blocks that upsample the image. We can also see a residual connection that takes the summation of the last encoder residual block and the last body residual block. Finally, we have two type-1 convolutional layers, i.e., the type-1 image convolutional layer and type-1 mask convolutional layer, that will produce a high-resolution (HR) face image and semantic segmentation map for the corresponding LR face image, respectively.  We can also see how the number of channels transforms between layers from the label written on the bottom of the layer with the format "\<previous number of channels\> to \<next number of channels\>".

FPN behaves in a multi-task setting to produce the segmentation map and HR face image at the same time for the corresponding LR face image. However, PSFRGAN will only use the semantic segmentation map for future prediction. The generation of the HR face image behaves as supervision for predicting a better semantic segmentation map.

Each residual block in FPN consists of 3 convolutional layers with different types according to which part of the network it belongs to, i.e., encoder residual block, body residual block, or decoder residual block. The detailed architecture of the residual block is shown below. The input goes to convolutional layer 1 and convolutional layer 2. From convolutional layer 2, it is propagated to convolutional layer 3. Finally, we take the summation of the output from convolutional layer 1 and convolutional layer 3 as the final output of the residual block.

![res-block.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/ebea0304-80a1-8c69-9b06-5ff6da8a86e8.png)

The table below shows the type for each convolutional layer for each kind of residual block.

convolutional layer|encoder residual block|body residual block|decoder residual block|
----------------------|---------|---------|---------|
conv. layer 1|type-1|type-1|type-2|
conv. layer 2|type-3|type-3|type-4| 
conv. layer 3|type-5|type-5|type-5|

Finally, we will show the detailed components of each type of convolutional layer. Type-4 convolutional layer has a complete architecture that consists of 2x interpolation, 2D reflection padding, 2D convolution, batch normalization, and leaky ReLU. Meanwhile, type-1, type-2, type-3, and type-5 are just subsets of type-4.

![conv-layer.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/aa5fbd4c-f53b-2fc2-c7af-2ff964851c05.png)

#### Generator

The generator network generates the super-resolution (SR) face image given the LR face image and its segmentation map produced by FPN previously. Below is the detailed architecture of the generator network. It starts with a constant $F_0$ with dimension $1024x16x16$. During the process, this constant will be modified to learn the SR face image based on the concatenation of the LR face image and semantic segmentation map, which we will call the LR+segmap image. We know that the LR face image has dimension $3x512x512$ and the segmentation map has dimension $19x512x512$ so the concatenation of those two (LR+segmap image) has dimension $22x512x512$.

![generator.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/fba44bd3-589d-884d-3ae7-e2723520a3b8.png)

The first part of the network consists of a 2D convolution and style transformation block, which produces $F_1$ with dimension $1024x16x16$. The second part consists of upsampling, 2D convolution, and style transformation block repeated five times. Each repetition produces $F_2$, $F_3$, $F_4$, $F_5$, and $F_6$. Finally, it employs 2D convolution as the final layer to transform the image to RGB and predict the SR face image. There are two residual connections that take the summation of the output of 2D convolution with the output of the style transformation block.

The detailed output shapes of the layers in the second part, i.e., upsampling, 2D convolution, and style transformation block for each repetition, can be seen below.

upsampling|2D convolution|style transformation block|output|
----------|--------------|--------------------------|------|
$1024x32x32$|$512x32x32$|$512x32x32$|$F_2$|
$512x64x64$|$256x64x64$|$256x64x64$|$F_3$|
$256x128x128$|$128x128x128$|$128x128x128$|$F_4$|
$128x256x256$|$64x256x256$|$64x256x256$|$F_5$|
$64x512x512$|$32x512x512$|$32x512x512$|$F_6$|

We will look deeper at the style transformation block since it is the main point of the proposed framework. The style transformation block takes and processes the LR+segmap image to update the input $F_i$. The image below shows the detailed architecture of the style transformation block. It consists of two combinations of spatially-adaptive (SPADE) normalization, leaky ReLU, and 2D convolution. We can also see a residual connection connecting the input $F_i$ and the output of this block. The SPADE normalization takes both the concatenated LR+segmap image and $F_i$. It processes the concatenated LR+segmap image to modify the input $F_i$, which we will see the detailed explanation soon.

![st-block.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/9b0ca9f6-9d11-7f74-139f-f02cadd49a6d.png)

The image below shows the detailed architecture of the SPADE normalization.  The normalization interpolates the LR+segmap image to have the same height and width as the input $F_i$. It then transforms it to a 2D convolution which maps the input channel from $22$ to $64$, and finally, it is normalized using leaky ReLU. After that, the output is convolved into two parts to produce the modulation parameters, i.e., gamma and beta, tensors with spatial dimensions. The produced gamma and beta are then multiplied and added to the normalized $F_i$. $F_i$ is normalized using instance normalization.

![spade-norm.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/4d6fc5ea-f9d3-7271-b4ff-8231d0927dee.png)

#### Discriminator

We use a discriminator network to generate the discriminator results for each concatenated HR image with ground truth semantic segmentation map, i.e., HR+segmap image and concatenated SR image produced by generator network with ground truth semantic segmentation map, i.e., SR+segmap image. They use the discriminator results to calculate the loss of PSFR-GAN. We will explain the detail of the losses in PSFR-GAN in the next section.

The image below shows the detailed architecture of the discriminator network. The network has three discriminators. Each discriminator has four main convolutional layers. Each discriminator's output will be saved as the final results. After feeding the input to a discriminator, the network downsamples the input using 2D average pooling before passing it to the next discriminator. 

![discriminator.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/a435eba1-d88f-56c0-38e1-f25f2eb2a30b.png)

Each N-layer discriminator consists of a type-1 convolutional layer, four type-2 convolutional layers, and finally, a final 2D convolution.

![n-layer.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/e9d3a90c-c275-c687-617c-74575f101af2.png)

The type-1 convolutional layer consists of 2D reflection padding, 2D convolution, and leaky ReLU. Meanwhile, the type-2 convolutional layer consists of 2D reflection padding, 2D convolution, 2D average pooling (for downsampling), instance normalization, and leaky ReLU.

![conv-layer-v2.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/bfc2c8f6-d259-d66a-3b67-6bdbe76dd0b6.png)

#### PSFR-GAN

Finally, PSFR-GAN unifies the three networks, i.e., FPN, generator, and discriminator. Given an LR face image, PSFR-GAN first utilizes pre-trained FPN to generate the semantic segmentation map of the corresponding LR face image. Given the LR face image and its segmentation map, the generator network will generate the SR face image. After that, the discriminator network will generate the discriminator results of the concatenation of SR face image and its segmentation map (SR+segmap image) and the concatenation of HR face image and its segmentation map (HR+segmap image). Note that the discriminator is present only in the training process where the segmentation map is provided in the training data. The discriminator results will be used to calculate some losses they utilize in this framework, i.e., feature matching loss, generator loss, and discriminator loss.

### Losses

There are two groups of losses. The first group consists of semantic-aware style loss, pixel loss, feature matching loss, and generator loss. The loss of the first group is formulated in this equation, where $\lambda$ is the weight for each loss. In the training process, we assign $\lambda_{ss}=100$, $\lambda_{pix}=10$, $\lambda_{fm}=10$, and $\lambda_{gen}=1$ following the original paper.

```math
L_1=\lambda_{ss}L_{ss}+\lambda_{pix}L_{pix}+\lambda_{fm}L_{fm}+\lambda_{gen}L_{gen}
```

Meanwhile, the second group consists of discriminator loss.

```math
L_2=L_{disc}
```

The PSFR-GAN is trained by minimizing both losses alternatively.

#### Semantic-Aware Style Loss
To calculate semantic-aware style loss (SS loss), first, we extract features from the predicted SR face image and ground truth HR face image using features from VGG19. We extract 0th-2nd layer of VGG19 to build first module, 3rd-7th layer to build second module, 8th-16th layer to build third module, 17th-25th layer to build fourth module, and finally 26th-34th layer to build fifth module. We feed each image to the module and get five features (one from each module) for each image. For each image we will use 3rd-5th features to calculate the loss, we denote this as $x\\_feat$ for features extracted from SR face image and $y\\_feat$ for features extracted from HR face image. For each feature in $x\\_feat$ and $y\\_feat$, we take the summation of the gram matrix in each region of the segmentation map. There are 19 regions in total. We calculate the mean squared error (MSE) between gram matrix $x\\_feat$ and gram matrix $y\\_feat$ for each feature and append the result to the total loss to get the SS loss.

The loss is formulated using this equation, where $\hat{I_H}$ is the SR face image, $I_H$ is the HR face image, $M_j$ is the segmentation map with label $j$, $\phi_i$ is feature of the corresponding image extracted from VGG19, and $G(\phi_i,M_j)$ is the gram matrix of feature $\phi_i$ in region $M_j$.
```math
L_{ss}=\sum^{5}_{i=3}\sum^{18}_{j=0}\,||G(\phi_i(\hat{I_H}), M_j)-G(\phi_i(I_H), M_j)||_2 \\
G(\phi_i,M_j)=\frac{(\phi_i \odot M_j)^T(\phi_i \odot M_j)}{\sum{M_j}+\epsilon} \\
\epsilon=10^{-8}
```

#### Pixel Loss and Feature Matching Los
The paper combines both pixel loss and feature matching loss to generate the reconstruction loss. We calculate the pixel loss between SR face image and HR image using `torch.nn.L1Loss()`. The pixel loss is formulated using this equation.

```math
L_{pix}=||\hat{I_H}-I_H||_1
```

We calculate the feature matching loss between SR+segmap image and HR+segmap image by feeding those images into the discriminator network that consists of 3 discriminators where each discriminator consists of 4 layers. We denote the features extracted by each discriminator from SR+segmap image as $x\\_feat$ and the features extracted by each discriminator from HR+segmap image as $y\\_feat$. For each discriminator, we calculate the feature matching by summation of the MSE between $x\\_feat$ and $y\\_feat$. Finally, we take the average loss from each discriminator. The feature matching loss is formulated using this equation.

```math
L_{fm}=\frac{1}{3}\sum^{3}_{s=1}\sum^{4}_{k=1}||D^k_s(\hat{I^s_H})-D^k_s(I^s_H)||_2
```

#### Generator Loss
We calculate the generator loss of the SR+segmap image by feeding the image into the discriminator network that consists of 3 discriminators. For each output of each discriminator, we calculate the loss by taking the negative mean of the output (hinge loss). We take the average loss from each discriminator. The generator loss is formulated using this equation.

```math
L_{gen}=\frac{1}{3}\sum^{3}_{s=1}-mean(\hat{I_H^s})
```

#### Discriminator Loss

We calculate the discriminator loss by feeding SR+segmap image and HR+segmap image into the discriminator network that consists of 3 discriminators. For each output of each discriminator, we calculate the loss from SR+segmap image by taking ReLU of the mean of $1+output$. Meanwhile, we calculate the loss from HR+segmap image by taking ReLU of the mean of $1-output$. We take the average loss from each discriminator. The discriminator loss is formulated using this equation.

```math
L_{disc}=0.5\frac{1}{N}\sum^{3}_{s=1}ReLU(mean(1-I_H^s))+ReLU(mean(1+\hat{I^s_H}))
```

### Training

#### <a name="train-dataset"></a> Dataset

We gather the dataset for training from FFHQ which is available at [github.com/NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset). We download the 70,000 images with $1024x1024$ resolution from [Google Drive](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL) using [rclone](https://rclone.org/). However, because of the limitation of training resources we only use 35,000 images.

We first apply random grayscale with probability of $0.3$.

```
import imgaug
from imgaug import augmenters

augmenters.Sometimes(0.3, augmenters.Grayscale(alpha=1.0))
```

We downsample the images to $512x512$ resolution to generate high-resolution (HR) face images. To generate the low-resolution (LR) face images, we feed the images to a degradation model. In the degradation model, we first apply a blur kernel $\mathbf{k}_\varrho$ with probability $0.5$. We randomly choose one between the four options: gaussian blur ($3 \le \varrho \le 15$), average blur ($3 \le \varrho \le 15$), median blur ($3 \le \varrho 15$), and motion blur ($5 \le \varrho \le 25$). Then, we downsample the image using the scale factor randomly chosen from $\frac{32}{512} \le r \le \frac{256}{512}$.

After that, we apply additive gaussian noise $\mathbf{n}_\delta$ ($0 \le \delta \le 0.1*255$) with probability $0.2$. We also apply JPEG compression with a compression rate randomly chosen from $10 \le r \le 65$ with a probability of $0.7$. Finally, we upsample the image to $512x512$ resolution. We summarize the code for the degradation model:

```
import imgaug
from imgaug import augmenters
import numpy as np

scale_size = np.random.randint(32, 256)
high_res_size = 512

augmenters.Sequential([
    augmenters.Sometimes(0.5, augmenters.OneOf([
        augmenters.GaussianBlur((3, 15)),
        augmenters.AverageBlur((3, 15)),
        augmenters.MedianBlur((3, 15)),
        augmenters.MotionBlur((5, 25))
    ])),
    augmenters.Resize(scale_size, interpolation=imgaug.ALL),
    augmenters.Sometimes(0.2, augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 25.5), per_channel=0.5)),
    augmenters.Sometimes(0.7, augmenters.JpegCompression(compression=(10, 65))),
    augmenters.Resize(high_res_size)
])
```

#### Setting

For generating the segmentation map for LQ face images and HQ face images, we use the pre-trained model provided by the original paper which is available here [parse_multi_iter_90000.pth](https://drive.google.com/drive/folders/1Ubejhxd2xd4fxGc_M_LWl3Ux6CgQd9rP). Due to the limitation of training resources, we train the model only with 20 epochs and batch size 4. The result models are provided here [trained_models](https://drive.google.com/drive/u/1/folders/12gAvnmVCKoJSPgAxZIbMcMm4IW15W2Fp). The script to train the model can be found at `train.py`. The config for this script can be changed from `config/base.json` and `config/psfrgan/train.json`.

### Testing

#### Dataset

We gather dataset from CelebAHQ which is available at [github.com/tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans). We download the images from [Google Drive](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P) using [rclone](https://rclone.org/). We choose 2,800 images with $1024x1024$ resolution randomly and resize the original images to $512x512$ resolution to make the ground truth HR face images. We generate the LR face images from these ground truth images using the degradation model we have explained in the training section. The script to generate the LR face images is available at `downsample_psfrgan.py`. The config for this script can be changed from `config/downsample.json`. 

#### Evaluation

We use the model [epoch_20_net_gen.pth](https://drive.google.com/drive/u/1/folders/12gAvnmVCKoJSPgAxZIbMcMm4IW15W2Fp) to generate super-resolution (SR) face images from LR face images. We measure Peak Signal-To-Noise Ratio (PSNR), Learned Perceptual Image Patch Similarity (LPIPS), Structural Similarity Index Measure (SSIM), and Fréchet Inception Distance (FID) between the predicted SR face images and ground truth HR face images. The original paper does not mention which libraries they use to evaluate the model. Therefore, we use libraries provided by [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/) to measure [PSNR](https://torchmetrics.readthedocs.io/en/v0.8.2/image/peak_signal_noise_ratio.html), [LPIPS](https://torchmetrics.readthedocs.io/en/v0.8.2/image/learned_perceptual_image_patch_similarity.html), and [SSIM](https://torchmetrics.readthedocs.io/en/stable/image/structural_similarity.html). For measuring FID, we use the library provided here [github.com/mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid).

Note that we do not use the same dataset and metrics implementation as the original paper because they are not publicly available. Therefore, we do not compare the performance of our model (`ours_epoch20`) with the one written in the original paper. Instead, we compare our model with the model provided in the original repository [psfrgan_epoch15_net_G.pth](https://drive.google.com/drive/folders/1Ubejhxd2xd4fxGc_M_LWl3Ux6CgQd9rP), we denote this model as `original_epoch15`. However, we are unsure whether this is the same model reported in the paper. Note that `ours_epoch20` is trained using 35,000 images provided by FFHQ, meanwhile, `original_epoch15` is trained using 70,000 images provided by FFHQ.

model|PSNR$\uparrow$|SSIM$\uparrow$|LPIPS$\downarrow$|FID$\downarrow$|
-----|--------|----|------|---|
original_epoch15|24.8226|0.6729|0.3347|12.0026|
ours_epoch20|24.5472|0.6648|0.3290|11.9687|

We provide some examples of images generated by each model.

code|HR face image|LR face image|ours_epoch20|original_epoch15|
-----|--------|----|------|---|
02911.jpg|<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/b8edc90b-5ec3-fb12-d88e-b1b485943bde.jpeg" width="130">|<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/00ab7db1-40e7-4f61-4913-2e6ab3f06fc8.jpeg" width="130">|<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/9bfb8c77-afbd-0eb4-1f25-d9a8ebca37b7.jpeg" width="130">|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/f95a0f60-10c0-4ec2-ddd7-69318bde942c.jpeg">|
03277.jpg|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/cb7d8716-1cf2-69cd-e46d-5b2a17cd1ad4.jpeg">|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/78e0c6d6-5a53-6ae3-fb08-00db2a49cae6.jpeg">|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/468db0ac-6ee6-57fb-6847-7f4f56d870b5.jpeg">|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/fb433c17-a469-bab7-4042-9eaff8dfdcf2.jpeg">|
04957.jpg|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/494c717c-0d37-a578-91c7-545ae722cbd1.jpeg">|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/5d64b5b6-813a-2adf-2d51-0a5b9596aa50.jpeg">|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/68f9584d-4250-81d0-b22d-a938852ef6cd.jpeg">|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/09a50d85-c741-cad3-f69c-86031151fc48.jpeg">|
29819.jpg|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/1383edb5-8a82-24ba-e2bc-f3baca8686cf.jpeg">|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/58b8c587-0f58-145a-2db2-42869363a2ba.jpeg">|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/a86a222b-3ea7-3a17-21a9-da550800769a.jpeg">|<img width="130" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2739288/02dba185-d8d3-cd6e-43e2-038c58c2c029.jpeg">|

We can see from the examples that our model produces relatively sharper images, e.g., `04957.jpg` and `29819.jpg`. However, the model provided in the original repository produces relatively smoother images with fewer artifacts, especially if we take a look at `03277.jpg` within the lip area.

The script to predict the test dataset is available at `test_psfrgan.py`. The config for this script can be changed from `config/base.json` and `config/psfrgan/test.json`. Meanwhile, the script to calculate the metrics of the predicted images, i.e., PSNR, LPIPS, and SSIM, is available at `evaluate_psfrgan.py`. The config of this script can be changed from `config/evaluate.json`. To calculate FID score, we simply run `python -m pytorch_fid <path/to/sr-folder> <path/to/hr-folder>` in the terminal.

#### Try with Own Image

You can experiment with the model using your own real LR face images. The script to preprocess the real LR face images is available at `preprocess_psfrgan.py`. The config of this script can be changed from `config/preprocess.json`. The script will produce cropped and aligned versions of the original LR faces which are ready to be enhanced by the model.

## References

C. Chen, X. Li, L. Yang, X. Lin, L. Zhang and K. -Y. K. Wong, "Progressive Semantic-Aware Style Transformation for Blind Face Restoration," 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 11891-11900, doi: 10.1109/CVPR46437.2021.01172.

T. Karras, S. Laine and T. Aila, "A Style-Based Generator Architecture for Generative Adversarial Networks," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 4396-4405, doi: 10.1109/CVPR.2019.00453.

T. Karras, T. Aila, S. Laine, and J. Lehtinen. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196, 2017.

Maximilian Seitzer. 2020. pytorch-fid: FID Score for PyTorch. https://github.com/mseitzer/pytorch-fid. Version 0.2.1.
