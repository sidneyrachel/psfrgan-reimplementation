import torch
from torch import nn


class FMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x_feats, y_feats):
        loss = 0

        for x_feat, y_feat in zip(x_feats, y_feats):
            loss = loss + self.mse(x_feat, y_feat.detach())

        return loss
