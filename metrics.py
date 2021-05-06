from torch import nn
import torch


class EarthMoversDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # input has dims: (Batch x Bins)
        bins = x.size(1)
        r = torch.arange(bins)
        s, t = torch.meshgrid(r, r)
        tt = t >= s

        cdf_x = torch.matmul(x, tt.float())
        cdf_y = torch.matmul(y, tt.float())

        return torch.sum(torch.square(cdf_x - cdf_y), dim=1)


class MutualInformationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p1, p2, p12):
        # input p12 has dims: (Batch x Bins x Bins)
        # input p1 & p2 has dims: (Batch x Bins)

        product_p = torch.matmul(torch.transpose(p1.unsqueeze(1), 1, 2), p2.unsqueeze(1)) + torch.finfo(p1.dtype).eps
        mi = torch.sum(p12 * torch.log(p12 / product_p + torch.finfo(p1.dtype).eps), dim=(1, 2))
        h = -torch.sum(p12 * torch.log(p12 + torch.finfo(p1.dtype).eps), dim=(1, 2))

        return 1 - (mi / h)
