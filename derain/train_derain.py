"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: train.py
about: main entrance for training the GridDehazeNet
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import time
import mindspore

import argparse
import mindspore.nn as nn
import matplotlib.pyplot as plt

from val_data import ValData

from model import Net
from utils import validation
plt.switch_backend('agg')
from mindspore import context, ops, Tensor
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.transforms import Compose

from mindspore.dataset.vision import Normalize, ToTensor

context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-3, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[120, 120], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=8, type=int)
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (indoor or outdoor?)', default='dehaze-SR', type=str)
args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\n'
      'num_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #
num_epochs = 300
val_data_dir_dehaze = './datalists/test/dehaze/'
val_data_dir_deblur = './datalists/test/deblur/'
val_data_dir_denoise = './datalists/test/denoise/'
val_data_dir_derain = './datalists/test/derain/'
val_data_dir_SR = './datalists/test/SR/'

if category == 'dehaze-derain':
    train_data_dir = ['./datalists/train/dehaze/','./datalists/train/derain/']
elif category == 'dehaze-deblur':
    train_data_dir = ['./datalists/train/dehaze/','./datalists/train/deblur/']
elif category == 'dehaze-denoise':
    train_data_dir = ['./datalists/train/dehaze/','./datalists/train/denoise/']
elif category == 'dehaze-SR':
    train_data_dir = ['./datalists/train/dehaze/','./datalists/train/SR/']

# --- Define the network --- #
net = Net()



# --- Load training data and validation/test data --- #val_batch_size
val_data_loader_dehaze = ValData(val_data_dir_dehaze)
val_data_loader_deblur = ValData(val_data_dir_deblur)
val_data_loader_denoise = ValData(val_data_dir_denoise)
val_data_loader_derain = ValData(val_data_dir_derain)
val_data_loader_SR = ValData(val_data_dir_SR)

trans = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),is_hwc=False)])
val_data_loader_derain = GeneratorDataset(
        val_data_loader_derain, 
        column_names=['input','gt','id'], 
        shuffle=False,
    )
val_data_loader_derain = val_data_loader_derain.map(operations=trans,
                                input_columns=["input"])
val_data_loader_derain = val_data_loader_derain.batch(1, drop_remainder=True)

param_dict = mindspore.load_checkpoint('./ms_derain_derain_best.ckpt')
not_load_list = mindspore.load_param_into_net(net, param_dict)
old_val_psnr_derain, old_val_ssim_derain = validation(net, val_data_loader_derain, img_scale=False)
print('old_val_psnr_derain: {0:.2f}, old_val_ssim_derain: {1:.4f}'.format(old_val_psnr_derain, old_val_ssim_derain))


# old_val_psnr_derain, old_val_ssim_derain = validation(net, val_data_loader_derain, category)
# print('old_val_psnr_derain: {0:.2f}, old_val_ssim_derain: {1:.4f}'.format(old_val_psnr_derain, old_val_ssim_derain))


# old_val_psnr_deblur, old_val_ssim_deblur = validation(net, val_data_loader_deblur, category)
# print('old_val_psnr_deblur: {0:.2f}, old_val_ssim_deblur: {1:.4f}'.format(old_val_psnr_deblur, old_val_ssim_deblur))

# old_val_psnr_dehaze, old_val_ssim_dehaze = validation(net, val_data_loader_dehaze, category)
# print('old_val_psnr_dehaze: {0:.2f}, old_val_ssim_dehaze: {1:.4f}'.format(old_val_psnr_dehaze, old_val_ssim_dehaze))

# old_val_psnr_SR, old_val_ssim_SR = validation(net, val_data_loader_SR, category,img_scale=True)
# print('old_val_psnr_SR: {0:.2f}, old_val_ssim_SR: {1:.4f}'.format(old_val_psnr_SR, old_val_ssim_SR))
