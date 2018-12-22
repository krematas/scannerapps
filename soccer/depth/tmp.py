import scannerpy
import numpy as np
import os
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType

from os.path import join
import glob

import torch
import torch.nn as nn
from torchvision import transforms
from typing import Sequence

from soccer.depth.hourglass import hg8
import matplotlib.pyplot as plt
from scipy.misc import imresize

import argparse
import time
import subprocess as sp
import cv2

# Testing settings
parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/emil-forsberg-goal-sweden-v-switzerland-match-55')
parser.add_argument('--path_to_model', default='/home/krematas/Mountpoints/grail/tmp/cnn/model.pth')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--cloud', action='store_true')
parser.add_argument('--bucket', default='', type=str)

opt, _ = parser.parse_known_args()


dataset = opt.path_to_data
image_files = glob.glob(join(dataset, 'players', 'images', '*.jpg'))
mask_files = glob.glob(join(dataset, 'players', 'masks', '*.png'))


mask = []
image = []
for i in range(10):
    img = cv2.imread(image_files[i])[:, :, ::-1]
    image.append(img)
    mask.append(cv2.imread(mask_files[i]))


normalize = transforms.Normalize(mean=[0.3402085, 0.42575407, 0.23771574],
                                             std=[0.1159472, 0.10461029, 0.13433486])
img_size = 256
batch = torch.zeros([10, 4, img_size, img_size], dtype=torch.float32)

for i in range(10):

    _image = imresize(image[i], (img_size, img_size))
    _mask = imresize(mask[i][:, :, 0], (img_size, img_size), interp='nearest', mode='F')

    # ToTensor
    _image = _image.transpose((2, 0, 1))/255.0
    _mask = _mask[:, :, None].transpose((2, 0, 1))/255.0

    image_tensor = torch.from_numpy(_image)
    image_tensor = torch.FloatTensor(image_tensor.size()).copy_(image_tensor)
    image_tensor = normalize(image_tensor)
    mask_tensor = torch.from_numpy(_mask)

    batch[i, :, :, :] = torch.cat((image_tensor, mask_tensor), 0)
# Normalize
batch = batch.cuda()
# Make it BxCxHxW
