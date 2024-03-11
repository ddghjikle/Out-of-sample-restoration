import os
import re
import sys

def change_images_name(path,index = 9):
    fileList = os.listdir(path)
    print("before:" + str(fileList))
    os.chdir(path)
    for fileName in fileList:
        pat = ".+\.(jpg|jpeg|JPG)"
        pattern = re.findall(pat, fileName)
        print('pattern[0]:', pattern,'filename:',fileName[0:index])
        os.rename(fileName,(fileName[0:index] + '.' + pattern[0]))
    sys.stdin.flush()
    print("after:" + str(os.listdir(path)))
    print("---------------------------------------------------")

# indoor
path = "/root/projects/GridDehazeNet-master/ihazeohazedatasets/i-haze/test/clear/"
change_images_name(path,index=9)

path = "/root/projects/GridDehazeNet-master/ihazeohazedatasets/i-haze/test/hazy/"
change_images_name(path,index=9)

path = "/root/projects/GridDehazeNet-master/ihazeohazedatasets/i-haze/train/clear/"
change_images_name(path,index=9)

path = "/root/projects/GridDehazeNet-master/ihazeohazedatasets/i-haze/train/hazy/"
change_images_name(path,index=9)

# outdoor
path = "/root/projects/GridDehazeNet-master/ihazeohazedatasets/o-haze/test/clear/"
change_images_name(path,index=10)

path = "/root/projects/GridDehazeNet-master/ihazeohazedatasets/o-haze/test/hazy/"
change_images_name(path,index=10)

path = "/root/projects/GridDehazeNet-master/ihazeohazedatasets/o-haze/train/clear/"
change_images_name(path,index=10)

path = "/root/projects/GridDehazeNet-master/ihazeohazedatasets/o-haze/train/hazy/"

change_images_name(path,index=10)

def show_images_list(images_path='',save_list_path='',file_name=''):

    fileList = sorted(os.listdir(images_path))
    with open(save_list_path+file_name, 'w') as file_handle:
        for fileName in fileList:
            file_handle.write("{}\n".format(fileName))

# indoor
images_path = "/root/projects/GridDehazeNet-master/ihazeohazedatasets/i-haze/train/hazy/"
save_list_path='/root/projects/GridDehazeNet-master/ihazeohazedatasets/i-haze/train/'
file_name='train_list.txt'
show_images_list(images_path,save_list_path,file_name)

images_path = "/root/projects/GridDehazeNet-master/ihazeohazedatasets/i-haze/test/hazy/"
save_list_path='/root/projects/GridDehazeNet-master/ihazeohazedatasets/i-haze/test/'
file_name='test_list.txt'
show_images_list(images_path,save_list_path,file_name)

# outdoor
images_path = "/root/projects/GridDehazeNet-master/ihazeohazedatasets/o-haze/train/hazy/"
save_list_path='/root/projects/GridDehazeNet-master/ihazeohazedatasets/o-haze/train/'
file_name='train_list.txt'
show_images_list(images_path,save_list_path,file_name)

images_path = "/root/projects/GridDehazeNet-master/ihazeohazedatasets/o-haze/test/hazy/"
save_list_path='/root/projects/GridDehazeNet-master/ihazeohazedatasets/o-haze/test/'
file_name='test_list.txt'
show_images_list(images_path,save_list_path,file_name)

from PIL import Image
def change_images_size(image_path):
    fileList = sorted(os.listdir(image_path))
    scale = 1.0/ 8
    for fileName in fileList:
        print(fileName)
        img = Image.open(image_path+fileName)
        new_size= (img.size[0]*scale, img.size[1]*scale)
        img.thumbnail(new_size, Image.BILINEAR)
        img.save(image_path+fileName, 'jpeg')
    print('--------------------------------------------')

def change_images_size_V2(image_path):
    fileList = sorted(os.listdir(image_path))
    for fileName in fileList:
        print(fileName)
        img = Image.open(image_path+fileName)
        # new_size= (640*2, 480*2)
        new_size= (640, 480)
        img = img.resize(new_size, Image.BILINEAR)
        img.save(image_path+fileName, 'jpeg')
    print('--------------------------------------------')


# # indoor
# path = "/root/projects/DualResidualNetworks-master/ihazeohazedatasets/i-haze/train/images/"
# change_images_size_V2(path)
#
# path = "/root/projects/DualResidualNetworks-master/ihazeohazedatasets/i-haze/train/labels/"
# change_images_size_V2(path)
#
# path = "/root/projects/DualResidualNetworks-master/ihazeohazedatasets/i-haze/test/images/"
# change_images_size_V2(path)
#
# path = "/root/projects/DualResidualNetworks-master/ihazeohazedatasets/i-haze/test/labels/"
# change_images_size_V2(path)
#
#
# # outdoor
# path = "/root/projects/DualResidualNetworks-master/ihazeohazedatasets/o-haze/train/images/"
# change_images_size_V2(path)
#
# path = "/root/projects/DualResidualNetworks-master/ihazeohazedatasets/o-haze/train/labels/"
# change_images_size_V2(path)
#
# path = "/root/projects/DualResidualNetworks-master/ihazeohazedatasets/o-haze/test/images/"
# change_images_size_V2(path)
#
# path = "/root/projects/DualResidualNetworks-master/ihazeohazedatasets/o-haze/test/labels/"
# change_images_size_V2(path)
