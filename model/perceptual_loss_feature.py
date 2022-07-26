import torch
from torch import nn
from torchvision import models

from variable.model import PerceptualLossBaseModelEnum


# Features used to calculate Perceptual Loss based on ResNet50 features.
class PerceptualLossFeature(nn.Module):
    def __init__(
            self,
            weight_path,
            model_name=PerceptualLossBaseModelEnum.VGG_19.value
    ):
        super(PerceptualLossFeature, self).__init__()
        self.features = None

        if model_name == PerceptualLossBaseModelEnum.VGG_19.value:
            self.model = models.vgg19(pretrained=False)
            self.build_vgg_layers()
        elif model_name == PerceptualLossBaseModelEnum.RESNET_50.value:
            self.model = models.resnet50(pretrained=False)
            self.build_resnet_layers()
        else:
            raise Exception(f'Base model is not supported. Model name: {model_name}.')

        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def build_resnet_layers(self):
        self.layer1 = torch.nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1
        )
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

        self.features = torch.nn.ModuleList([
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        ])

    def build_vgg_layers(self):
        vgg_pretrained_features = self.model.features
        features = []
        feature_layers = [0, 3, 8, 17, 26, 35]

        for i in range(len(feature_layers) - 1):
            module_layers = torch.nn.Sequential()

            for j in range(feature_layers[i], feature_layers[i + 1]):
                module_layers.add_module(str(j), vgg_pretrained_features[j])

            features.append(module_layers)

        self.features = torch.nn.ModuleList(features)

    def preprocess(self, inp):
        outp = (inp + 1) / 2
        # Value explanation:
        # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(outp)  # Change dtype to match output
        std = torch.Tensor([0.229, 0.224, 0.225]).to(outp)  # Change dtype to match output
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        outp = (outp - mean) / std

        if outp.shape[3] < 224:
            outp = torch.nn.functional.interpolate(outp, size=(224, 224), mode='bilinear', align_corners=False)

        return outp

    def forward(self, inp):
        outp = self.preprocess(inp)

        features = []

        for layer in self.features:
            outp = layer(outp)
            features.append(outp)

        return features
