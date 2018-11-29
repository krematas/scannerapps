import scannerpy
from scannerpy import Database, Job, FrameType
import cv2
import glob
from os.path import join, basename
from skimage.morphology import medial_axis
import os

import numpy as np
import time
import soccer.calibration.utils as utils
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--cloud', action='store_true')
parser.add_argument('--bucket', default='', type=str)
parser.add_argument('--nframes', type=int, default=5, help='Margin around the pose')
parser.add_argument('--framestep', type=int, default=10, help='Margin around the pose')
parser.add_argument('--nworkers', type=int, default=5, help='Margin around the pose')
parser.add_argument('--work_packet_size', type=int, default=4, help='Margin around the pose')
parser.add_argument('--io_packet_size', type=int, default=8, help='Margin around the pose')


opt, _ = parser.parse_known_args()


path_to_data = opt.path_to_data

goal_dirs = [item for item in os.listdir(path_to_data) if os.path.isdir(os.path.join(path_to_data, item)) ]
goal_dirs.sort()

goal_id = 4
dataset = join(path_to_data,goal_dirs[goal_id])
print('Processing dataset: {0}'.format(dataset))

h, w = 1080, 1920


master = 'localhost:5001'
workers = ['localhost:{:d}'.format(d) for d in range(5002, 5002+opt.nworkers)]
# db = Database(master=master, workers=workers)
db = Database()

config = db.config.config['storage']
params = {'bucket': opt.bucket,
          'storage_type': config['type'],
          'endpoint': 'storage.googleapis.com',
          'region': 'US'}

image_files_all = glob.glob(join(dataset, 'images', '*.jpg'))
image_files_all.sort()

mask_files_all = glob.glob(join(dataset, 'detectron', '*.png'))
mask_files_all.sort()

cam_data = np.load(join(dataset, 'calib', '{0}.npy'.format(basename(image_files_all[0])[:-4]))).item()


image_files = [image_files_all[i] for i in range(0, len(image_files_all), opt.framestep)]
mask_files = [mask_files_all[i] for i in range(0, len(mask_files_all), opt.framestep)]
indeces = [i for i in range(0, len(image_files_all), opt.framestep)]

if image_files_all[-1] not in image_files:
    image_files.append(image_files_all[-1])
    mask_files.append(mask_files_all[-1])
    indeces.append(len(image_files_all)-1)

encoded_image = db.sources.Files(**params)
frame_img = db.ops.ImageDecoder(img=encoded_image)

encoded_mask = db.sources.Files(**params)
frame_mask = db.ops.ImageDecoder(img=encoded_mask)


cwd = '/home/krematas/code/scannerapps/soccer/calibration/'
db.load_op(os.path.join(cwd, 'distancetr_op/build/libdistancetr_op.so'), os.path.join(cwd, 'distancetr_op/build/distance_pb2.py'))

dist_transform_class = db.ops.DistanceTransform(frame=frame_img, mask=frame_mask, h=h, w=w)
output_op = db.sinks.FrameColumn(columns={'frame': dist_transform_class})

job = Job(
    op_args={
    encoded_image: {'paths': image_files, **params},
    encoded_mask: {'paths': mask_files, **params},
    output_op: 'example_resized55',
})


start = time.time()
[out_table] = db.run(output_op, [job], force=True, pipeline_instances_per_node=1,
                     work_packet_size=opt.work_packet_size, io_packet_size=opt.io_packet_size)
end = time.time()
print('scanner distance transform: {0:.4f}'.format(end-start))

results = out_table.column('frame').load()

for res in results:
    plt.imshow(res)
    plt.show()
    # break