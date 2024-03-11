"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: utils.py
about: all utilities
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import time
import mindspore
import torchvision.utils as utils
from math import log10
from skimage import measure
from mindspore.dataset.transforms import Compose

import mindspore.dataset.vision as vision
def to_psnr(dehaze, gt):
    mse = mindspore.ops.mse_loss(dehaze, gt, reduction='none')
    mse_split = mindspore.ops.split(mse, 1, axis=0)
    mse_list = [mindspore.ops.mean(mindspore.ops.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = mindspore.ops.split(dehaze, 1, axis=0)
    gt_list = mindspore.ops.split(gt, 1, axis=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list
from mindspore import Tensor
def validation(net, val_data_loader, img_scale=False):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []
    to_tensor = vision.ToTensor()

    for batch_id, val_data in enumerate(val_data_loader):

        haze, gt, image_name = val_data

        gt = Tensor(to_tensor(gt.numpy().squeeze())).unsqueeze(dim=0)
        if img_scale:
            dehaze = net(haze,[gt.shape[-2],gt.shape[-1]])
        else:
            dehaze = net(haze)

        output_images=mindspore.Tensor.chunk(dehaze,dehaze.shape[0],axis=0)
        gt_images=mindspore.Tensor.chunk(gt,gt.shape[0],axis=0)
        for i in range(dehaze.shape[0]):
        # --- Calculate the average PSNR --- #
            psnr_list.extend(to_psnr(output_images[i], gt_images[i]))

        # --- Calculate the average SSIM --- #
            ssim_list.extend(to_ssim_skimage(output_images[i], gt_images[i]))
        print(psnr_list)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim



