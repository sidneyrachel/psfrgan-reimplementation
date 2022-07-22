import torch
from torch import nn


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.mse = torch.nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x_feats, y_feats):
        loss = 0
        for x_feat, y_feat, weight in zip(x_feats, y_feats, self.weights):
            loss = loss + self.mse(x_feat, y_feat.detach()) * weight

        return loss
