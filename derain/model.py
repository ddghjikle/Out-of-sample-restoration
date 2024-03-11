# -*- coding: utf-8 -*- 
from cmath import phase
import functools
import numpy as np

import mindspore
import mindspore.nn as nn
import math


import pdb

class _Residual_Block(nn.Cell): 
    def __init__(self,in_channels):
        super(_Residual_Block, self).__init__()
        self.in_channels=in_channels
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1,pad_mode='pad')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1,pad_mode='pad')
        

    def construct(self, x): 
        identity_data = x
        output = self.relu(self.conv1(x))
        output=self.conv2(output)
        output = mindspore.ops.add(output,identity_data)
        return output 

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_input_3= nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3, padding=1,pad_mode='pad',has_bias=True)

        self.res_1=_Residual_Block(in_channels=16)
        self.res_2=_Residual_Block(in_channels=16)
        self.conv_up_1=nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3, padding=1,pad_mode='pad',has_bias=True)
        self.res_3=_Residual_Block(in_channels=64)
        self.res_4=_Residual_Block(in_channels=64)
        self.conv_up_2=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1,pad_mode='pad',has_bias=True)
        self.res_5=_Residual_Block(in_channels=64)
        self.res_6=_Residual_Block(in_channels=64)
      ###############上面为encoder,下面为decoder
        
        self.res_7=_Residual_Block(in_channels=64)
        self.res_8=_Residual_Block(in_channels=64)
        self.conv_down_1=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1,pad_mode='pad',has_bias=True)
        self.res_9=_Residual_Block(in_channels=64)
        self.res_10=_Residual_Block(in_channels=64)
        self.conv_down_2=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3, padding=1,pad_mode='pad',has_bias=True)
        self.res_11=_Residual_Block(in_channels=32)
        self.res_12=_Residual_Block(in_channels=32)
        self.conv_out_3=nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3, padding=1,pad_mode='pad',has_bias=True)

    def construct(self,input,target_size=None):
        if target_size:
            input=mindspore.ops.interpolate(input,target_size)

        output= self.conv_input_3(input)
        output=self.res_2(self.res_1(output))
        output=self.conv_up_1(output)

        output=self.res_4(self.res_3(output))
        output=self.conv_up_2(output)

        output=self.res_6(self.res_5(output))
        output=self.res_8(self.res_7(output))
        output=self.conv_down_1(output)

        output=self.res_10(self.res_9(output))
        output=self.conv_down_2(output)

        output=self.res_12(self.res_11(output))
        output=self.conv_out_3(output)
        return output

