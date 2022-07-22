import torch
from torch import nn


class RegionStyleLoss(nn.Module):
    def __init__(self, num_region=19, eps=1e-8):
        super().__init__()
        self.num_region = num_region
        self.eps = eps
        self.mse = nn.MSELoss()

    def __masked_gram_matrix(self, inp, mask):
        batch_size, channel, height, width = inp.shape
        mask = mask.view(batch_size, -1, height * width)
        inp = inp.view(batch_size, -1, height * width)
        mask_sum = mask.sum(2) + self.eps

        inp = inp * mask
        base_matrix = torch.bmm(inp, inp.transpose(1, 2))
        return base_matrix / (channel * mask_sum.view(batch_size, 1, 1))

    def __layer_gram_matrix(self, inp, mask):
        gram_matrices = []

        for i in range(self.num_region):
            sub_mask = mask[:, i].unsqueeze(1)
            gram_matrix = self.__masked_gram_matrix(inp, sub_mask)
            gram_matrices.append(gram_matrix)

        return torch.stack(gram_matrices, dim=1)

    def forward(self, x_feats, y_feats, mask):
        loss = 0

        for x_feat, y_feat in zip(x_feats[2:], y_feats[2:]):
            tmp_mask = torch.nn.functional.interpolate(mask, x_feat.shape[2:])
            x_feat_gram_matrix = self.__layer_gram_matrix(x_feat, tmp_mask)
            y_feat_gram_matrix = self.__layer_gram_matrix(y_feat, tmp_mask)
            tmp_loss = self.mse(x_feat_gram_matrix, y_feat_gram_matrix.detach())
            loss = loss + tmp_loss

        return loss
