from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

from hist_layers import SingleDimHistLayer, JointHistLayer
from metrics import EarthMoversDistanceLoss, MutualInformationLoss


def np_to_torch(img, add_batch_dim=True):
    img = np.asarray(img)
    img = img.astype(np.float32).transpose((2, 0, 1))
    if add_batch_dim:
        img = img[np.newaxis, ...]
    img_torch = torch.from_numpy(img) / 255.0
    return img_torch


def torch_to_np(tensor):
    tensor = tensor.detach().squeeze().cpu().numpy()
    if len(tensor.shape) < 3:
        return tensor
    else:
        return tensor.transpose(1, 2, 0)


def main():
    # TODO: convert RGB to YUV space
    # and sum loss for all channels

    result = np_to_torch(Image.open("deep_hist.png").resize((460, 460)).convert("RGB"))
    source = np_to_torch(Image.open("source.png").resize((460, 460)).convert("RGB"))
    target = np_to_torch(Image.open("target.png").resize((460, 460)).convert("RGB"))

    hist1 = SingleDimHistLayer()(source[:, 0])
    hist2 = SingleDimHistLayer()(target[:, 0])
    hist3 = SingleDimHistLayer()(result[:, 0])

    print("emd: source - target", EarthMoversDistanceLoss()(hist1, hist2))
    print("emd: target - result", EarthMoversDistanceLoss()(hist3, hist2))

    # we compare the differentiable histogram with the one produced by numpy
    _, ax = plt.subplots(2)
    ax[0].plot(hist1[0].cpu().numpy())
    ax[1].plot(np.histogram(source[:, 0].view(-1).cpu().numpy(), bins=256)[0])
    plt.show()

    joint_hist1 = JointHistLayer()(source[:, 0], result[:, 0])
    joint_hist2 = JointHistLayer()(target[:, 0], result[:, 0])
    joint_hist_self = JointHistLayer()(source[:, 0], source[:, 0])

    print("mi loss: source - result", MutualInformationLoss()(hist1, hist3, joint_hist1))
    print("mi loss: target - result", MutualInformationLoss()(hist2, hist3, joint_hist2))
    print("mi loss: source - source", MutualInformationLoss()(hist1, hist1, joint_hist_self))


if __name__ == '__main__':
    main()
