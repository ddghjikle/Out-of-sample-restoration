"""
paper: GridDeinputNet: Attention-Based Multi-Scale Network for Image Dehazing
file: val_data.py
about: build the validation/test dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
from mindspore.dataset import GeneratorDataset as data
from PIL import Image


# --- Validation/test dataset --- #
class ValData():
    def __init__(self, val_data_dir):
        super().__init__()
        if 'deblur' in  val_data_dir:
            val_input_list = val_data_dir + 'GOPRO_test_blur_file.txt'
            val_GT_list = val_data_dir + 'GOPRO_test_sharp_file.txt'
        if 'dehaze' in  val_data_dir:
            val_input_list = val_data_dir + 'SOTS_indoor_test_hazy.txt'
            val_GT_list = val_data_dir + 'SOTS_indoor_test_clear.txt'
        if 'denoise' in  val_data_dir:
            val_input_list = val_data_dir + 'CBSD68_clear.txt'
            val_GT_list = val_data_dir + 'CBSD68_noise.txt'
        if 'derain' in  val_data_dir:
            val_input_list = val_data_dir + 'Rain1200_test_rain_file.txt'
            val_GT_list = val_data_dir + 'Rain1200_test_norain_file.txt'
        if 'SR' in  val_data_dir:
            val_input_list = val_data_dir + 'B100_test_LR_file.txt'
            val_GT_list = val_data_dir + 'B100_test_HR_file.txt'

        
        with open(val_input_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            
        with open(val_GT_list) as f:
            contents = f.readlines()
            gt_names = [i.strip() for i in contents]

        self.input_names = input_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        input_img = Image.open(input_name).convert('RGB')
        gt_img = Image.open(gt_name).convert('RGB')



        return input_img, gt_img, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)