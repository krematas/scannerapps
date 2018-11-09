from scannerpy import Database, Job
import os.path
import glob
import argparse
from os.path import join
import pickle
import cv2
import soccer.kernels_c.resize_op.build.resize_pb2 as resize_pb2
import numpy as np


parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona/')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--cloud', action='store_true')
parser.add_argument('--bucket', default='', type=str)
parser.add_argument('--nframes', type=int, default=5, help='Margin around the pose')
opt, _ = parser.parse_known_args()

dataset = opt.path_to_data
total_files = 5
################################################################################
# This tutorial shows how to write and use your own C++ custom op.             #
################################################################################

# Look at resize_op/resize_op.cpp to start this tutorial.

db = Database()

config = db.config.config['storage']
params = {'bucket': opt.bucket,
          'storage_type': config['type'],
          'endpoint': 'storage.googleapis.com',
          'region': 'US'}

# ======================================================================================================================
# Images
# ======================================================================================================================
image_files = glob.glob(join(dataset, 'images', '*.jpg'))
image_files.sort()
image_files = image_files[:total_files]

encoded_image = db.sources.Files(**params)
frame = db.ops.ImageDecoder(img=encoded_image)


# ======================================================================================================================
# Instances
# ======================================================================================================================
file_to_save = '/home/krematas/Desktop/tmp.p'
with open(file_to_save, 'rb') as f:
    results = pickle.load(f)


data = []
for i, res in enumerate(results):
    buff = pickle.loads(res)
    for sel in range(len(buff)):
        h, w = buff[sel]['img'].shape[:2]

        _img = resize_pb2.MyImage()
        _, buffer = cv2.imencode('.jpg', buff[sel]['img'].astype(np.float32))
        _img.image_data = bytes(buffer)

        data.append([_img.SerializeToString()])

        if i == 0:
            break

data = data[:10]
print(len(data))
db.new_table('test', ['img'], data, force=True)


img = db.sources.FrameColumn()


cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.isfile(os.path.join(cwd, 'resize_op/build/libresize_op.so')):
    print(
        'You need to build the custom op first: \n'
        '$ pushd {}/resize_op; mkdir build && cd build; cmake ..; make; popd'.
        format(cwd))
    exit()

# To load a custom op into the Scanner runtime, we use db.load_op to open the
# shared library we compiled. If the op takes arguments, it also optionally
# takes a path to the generated python file for the arg protobuf.
db.load_op(
    os.path.join(cwd, 'resize_op/build/libresize_op.so'),
    os.path.join(cwd, 'resize_op/build/resize_pb2.py'))

# Then we use our op just like in the other examples.
resize = db.ops.MyResize(frame=img, width=200, height=300)
output_op = db.sinks.Column(columns={'resized_frame': resize})
job = Job(op_args={
    img: db.table('test').column('img'),
    # encoded_image: {'paths': image_files, **params},
    output_op: 'example_resized',
})
db.run(output_op, [job], force=True)