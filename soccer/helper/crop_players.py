import scannerpy
import cv2
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType

from os.path import join
import numpy as np
import glob
import argparse
import pickle

import time
import cocoapi.PythonAPI.pycocotools.mask as mask_util
from kernels import *

parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona/')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--cloud', action='store_true')
parser.add_argument('--bucket', default='', type=str)
opt, _ = parser.parse_known_args()


dataset = opt.path_to_data


db = Database()

config = db.config.config['storage']
params = {'bucket': opt.bucket,
          'storage_type': config['type'],
          'endpoint': 'storage.googleapis.com',
          'region': 'US'}


# ======================================================================================================================
# Images
image_files = glob.glob(join(dataset, 'images', '*.jpg'))
image_files.sort()

encoded_image = db.sources.Files(**params)
frame = db.ops.ImageDecoder(img=encoded_image)


# ======================================================================================================================
# Masks
mask_files = glob.glob(join(dataset, 'detectron', '*.png'))
mask_files.sort()

encoded_mask = db.sources.Files(**params)
mask_frame = db.ops.ImageDecoder(img=encoded_mask)


# ======================================================================================================================
# Metadata
with open(join(dataset, 'metadata', 'poses.p'), 'rb') as f:
    openposes = pickle.load(f)

frame_names = list(openposes.keys())
frame_names.sort()

pose_data = []
for fname in frame_names:
    n_poses = len(openposes[fname])
    poses_in_frame = np.zeros((n_poses, 18, 3), dtype=np.float32)
    for i in range(n_poses):
        poses_in_frame[i, :, :] = openposes[fname][i]
    data = {'poses': poses_in_frame}
    pose_data.append(data)

data = db.sources.Python()
pass_data = db.ops.Pass(input=data)


# ======================================================================================================================
# Scanner calls
draw_poses_class = db.ops.CropPlayersClass(image=frame, mask=mask_frame, metadata=pass_data, h=2160, w=3840)
output_op = db.sinks.FrameColumn(columns={'frame': draw_poses_class})

job = Job(
    op_args={
        encoded_image: {'paths': image_files, **params},
        encoded_mask: {'paths': mask_files},
        data: {'data': pickle.dumps(pose_data)},
        output_op: 'example_resized55',
    })

start = time.time()
[out_table] = db.run(output_op, [job], force=True)
end = time.time()


print('Total time for pose drawing in scanner: {0:.3f} sec'.format(end-start))
out_table.column('frame').save_mp4(join(dataset, 'players', 'poses'))
