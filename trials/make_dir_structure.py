import argparse
import os
from os.path import join


def mkdir(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)


parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/emil-forsberg-goal-sweden-v-switzerland-match-55')
opt, _ = parser.parse_known_args()

dataset = opt.path_to_data
mkdir(dataset)
mkdir(join(dataset, 'images'))
mkdir(join(dataset, 'detectron'))
mkdir(join(dataset, 'metadata'))
mkdir(join(dataset, 'calib'))
