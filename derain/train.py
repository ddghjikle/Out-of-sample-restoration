"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: train.py
about: main entrance for training the GridDehazeNet
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
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


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-3, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[240, 240], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=16, type=int)
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (indoor or outdoor?)', default='derain', type=str)
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

if category == 'dehaze':
    train_data_dir = './datalists/train/dehaze/'
elif category == 'deblur':
    train_data_dir = './datalists/train/deblur/'
elif category == 'denoise':
    train_data_dir = './datalists/train/denoise/'
elif category == 'derain':
    train_data_dir = './datalists/train/derain/'
elif category == 'SR':
    train_data_dir = './datalists/train/SR/'


# --- Gpu device --- #
device_ids = [1]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = Net()


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)
print(next(net.parameters()).device)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()
print(next(loss_network.parameters()).device)


# --- Load the network weight --- #
try:
    net.load_state_dict(torch.load('{}_haze_best_{}_{}'.format(category, network_height, network_width)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')


# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


# --- Load training data and validation/test data --- #val_batch_size
train_data_loader = DataLoader(TrainDataOutdoor(crop_size, train_data_dir, category), batch_size=train_batch_size, shuffle=True, num_workers=4)
val_data_loader_dehaze = DataLoader(ValData(val_data_dir_dehaze), batch_size=val_batch_size, shuffle=False, num_workers=1)
val_data_loader_deblur = DataLoader(ValData(val_data_dir_deblur), batch_size=val_batch_size, shuffle=False, num_workers=1)
val_data_loader_denoise = DataLoader(ValData(val_data_dir_denoise), batch_size=val_batch_size, shuffle=False, num_workers=1)
val_data_loader_derain = DataLoader(ValData(val_data_dir_derain), batch_size=val_batch_size, shuffle=False, num_workers=1)
val_data_loader_SR = DataLoader(ValData(val_data_dir_SR), batch_size=val_batch_size, shuffle=False, num_workers=1)


# --- Previous PSNR and SSIM in testing --- #
old_val_psnr_derain, old_val_ssim_derain = validation(net, val_data_loader_derain, device, category)
print('old_val_psnr_derain: {0:.2f}, old_val_ssim_derain: {1:.4f}'.format(old_val_psnr_derain, old_val_ssim_derain))

old_val_psnr_denoise, old_val_ssim_denoise = validation(net, val_data_loader_denoise, device, category)
print('old_val_psnr_denoise: {0:.2f}, old_val_ssim_denoise: {1:.4f}'.format(old_val_psnr_denoise, old_val_ssim_denoise))

old_val_psnr_deblur, old_val_ssim_deblur = validation(net, val_data_loader_deblur, device, category)
print('old_val_psnr_deblur: {0:.2f}, old_val_ssim_deblur: {1:.4f}'.format(old_val_psnr_deblur, old_val_ssim_deblur))

old_val_psnr_dehaze, old_val_ssim_dehaze = validation(net, val_data_loader_dehaze, device, category)
print('old_val_psnr_dehaze: {0:.2f}, old_val_ssim_dehaze: {1:.4f}'.format(old_val_psnr_dehaze, old_val_ssim_dehaze))

old_val_psnr_SR, old_val_ssim_SR = validation(net, val_data_loader_SR, device, category,img_scale=True)
print('old_val_psnr_SR: {0:.2f}, old_val_ssim_SR: {1:.4f}'.format(old_val_psnr_SR, old_val_ssim_SR))

for epoch in range(1,num_epochs+1):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, category=category)

    for batch_id, train_data in enumerate(train_data_loader):

        haze, gt = train_data
        haze = haze.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        if category == 'SR':
            dehaze = net(haze,[gt.size(-2),gt.size(-1)])
        else:
            dehaze = net(haze)

        smooth_loss = F.smooth_l1_loss(dehaze, gt)
        perceptual_loss = loss_network(dehaze, gt)
        loss = smooth_loss + lambda_loss*perceptual_loss

        loss.backward()
        optimizer.step()


        if not (batch_id % 100):
            print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

    if epoch %10 == 0:
    # --- Use the evaluation model in testing --- #
        net.eval()

        val_psnr_SR, val_ssim_SR = validation(net, val_data_loader_SR, device, category,img_scale=True)
        print('val_psnr_SR: {0:.2f}, val_ssim_SR: {1:.4f}'.format(val_psnr_SR, val_ssim_SR))
        if val_psnr_SR >= old_val_psnr_SR:
           torch.save(net.state_dict(), '{}_SR_best'.format(category))
           old_val_psnr_SR = val_psnr_SR
           old_val_ssim_SR = val_ssim_SR

        val_psnr_derain, val_ssim_derain = validation(net, val_data_loader_derain, device, category)
        print('val_psnr_derain: {0:.2f}, val_ssim_derain: {1:.4f}'.format(val_psnr_derain, val_ssim_derain))
        if val_psnr_derain >= old_val_psnr_derain:
            torch.save(net.state_dict(), '{}_derain_best'.format(category))
            old_val_psnr_derain = val_psnr_derain
            old_val_ssim_derain = val_ssim_derain

        val_psnr_denoise, val_ssim_denoise = validation(net, val_data_loader_denoise, device, category)
        print('val_psnr_denoise: {0:.2f}, val_ssim_denoise: {1:.4f}'.format(val_psnr_denoise, val_ssim_denoise))
        if val_psnr_denoise >= old_val_psnr_denoise:
            torch.save(net.state_dict(), '{}_denoise_best'.format(category))
            old_val_psnr_denoise = val_psnr_denoise
            old_val_ssim_denoise = val_ssim_denoise

        val_psnr_deblur, val_ssim_deblur = validation(net, val_data_loader_deblur, device, category)
        print('val_psnr_deblur: {0:.2f}, val_ssim_deblur: {1:.4f}'.format(val_psnr_deblur, val_ssim_deblur))
        if val_psnr_deblur >= old_val_psnr_deblur:
            torch.save(net.state_dict(), '{}_deblur_best'.format(category))
            old_val_psnr_deblur = val_psnr_deblur
            old_val_ssim_deblur = val_ssim_deblur

        val_psnr_dehaze, val_ssim_dehaze = validation(net, val_data_loader_dehaze, device, category)
        print('val_psnr_dehaze: {0:.2f}, val_ssim_dehaze: {1:.4f}'.format(val_psnr_dehaze, val_ssim_dehaze))
        if val_psnr_dehaze >= old_val_psnr_dehaze:
            torch.save(net.state_dict(), '{}_dehaze_best'.format(category))
            old_val_psnr_dehaze = val_psnr_dehaze
            old_val_ssim_dehaze = val_ssim_dehaze

one_process_time = time.time() - start_time
print('Training end!')
print('The bset val_psnr_SR: {0:.2f}, The bset val_ssim_SR: {1:.4f}'.format(old_val_psnr_SR, old_val_ssim_SR))
print('The bset val_psnr_derain: {0:.2f}, The bset val_ssim_derain: {1:.4f}'.format(old_val_psnr_derain, old_val_ssim_derain))
print('The bset val_psnr_denoise: {0:.2f}, The bset val_ssim_denoise: {1:.4f}'.format(old_val_psnr_denoise, old_val_ssim_denoise))
print('The bset val_psnr_deblur: {0:.2f}, The bset val_ssim_deblur: {1:.4f}'.format(old_val_psnr_deblur, old_val_ssim_deblur))
print('The bset val_psnr_dehaze: {0:.2f}, The bset val_ssim_dehaze: {1:.4f}'.format(old_val_psnr_dehaze, old_val_ssim_dehaze))
