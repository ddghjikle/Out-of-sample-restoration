import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# from train_data import TrainData
from val_data import ValData
from train_data_outdoor import TrainDataOutdoor
from model import Net
from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
plt.switch_backend('agg')

import mindspore as ms

def pytorch2mindspore(default_file = 'dehaze-SR_deblur_best'):
    # read pth file
    par_dict = torch.load(default_file,map_location=torch.device('cpu'))
    params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        param_dict['name'] = name
        param_dict['data'] = ms.Tensor(parameter.numpy())
        params_list.append(param_dict)
    ms.save_checkpoint(params_list,  'ms_'+default_file+'.ckpt')
    
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


# --- Gpu device --- #
device_ids = [6]
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
# net = Net(is_pretrain=None)

# --- Multi-GPU --- #
# net = net.to(device)
# net = nn.DataParallel(net, device_ids=device_ids)
# print(next(net.parameters()).device)

# net.load_state_dict(torch.load('dehaze-SR_deblur_best'))
# print('--- weight loaded ---')

pytorch2mindspore('derain_deblur_best')
pytorch2mindspore('derain_dehaze_best')
pytorch2mindspore('derain_denoise_best')
pytorch2mindspore('derain_derain_best')
pytorch2mindspore('derain_SR_best')
