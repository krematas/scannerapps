import scannerpy
import cv2
import numpy as np
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType
from scannerpy.stdlib import pipelines
import time

from os.path import join
import glob
import os
import subprocess as sp

import argparse


parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona/')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--cloud', action='store_true')
parser.add_argument('--bucket', default='', type=str)
opt, _ = parser.parse_known_args()


dataset = opt.path_to_data

if opt.cloud:
    def get_paths(path):
        paths = sp.check_output('gsutil ls gs://{:s}/{:s}'.format(opt.bucket, path),
                                shell=True).strip().decode('utf-8')
        paths = paths.split('\n')
        prefix_len = len('gs://{:s}/'.format(opt.bucket))
        stripped_paths = [p[prefix_len:] for p in paths]
        return stripped_paths
    image_files = get_paths(join(dataset, 'players', 'images', '*.jpg'))
    mask_files = get_paths(join(dataset, 'players', 'poseimgs', '*.png'))
else:
    image_files = glob.glob(join(dataset, 'players', 'images', '*.jpg'))
    mask_files = glob.glob(join(dataset, 'players', 'poseimgs', '*.png'))

image_files.sort()
mask_files.sort()


if opt.cloud:
    print('Finding master IP...')
    ip = sp.check_output(
        '''
    kubectl get pods -l 'app=scanner-master' -o json | \
    jq '.items[0].spec.nodeName' -r | \
    xargs -I {} kubectl get nodes/{} -o json | \
    jq '.status.addresses[] | select(.type == "ExternalIP") | .address' -r
    ''',
        shell=True).strip().decode('utf-8')
    
    port = sp.check_output(
        '''
    kubectl get svc/scanner-master -o json | \
    jq '.spec.ports[0].nodePort' -r
    ''',
        shell=True).strip().decode('utf-8')

    master = '{}:{}'.format(ip, port)
    print(master)
    db = Database(master=master, start_cluster=False, config_path='./config.toml',
                  grpc_timeout=60)
else:
    db = Database()


cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.isfile(os.path.join(cwd, 'segment_op/build/libsegment_op.so')):
    print(
        'You need to build the custom op first: \n'
        '$ pushd {}/segment_op; mkdir build && cd build; cmake ..; make; popd'.
        format(cwd))
    exit()

# To load a custom op into the Scanner runtime, we use db.load_op to open the
# shared library we compiled. If the op takes arguments, it also optionally
# takes a path to the generated python file for the arg protobuf.
if opt.cloud:
    db.load_op(
        '/app/segment_op/build/libsegment_op.so',
        os.path.join(cwd, 'segment_op/build/segment_pb2.py'))
else:
    db.load_op(
        os.path.join(cwd, 'segment_op/build/libsegment_op.so'),
        os.path.join(cwd, 'segment_op/build/segment_pb2.py'))

model_path = '/home/krematas/code/scannerapps/model.yml.gz'

config = db.config.config['storage']
params = {'bucket': opt.bucket,
          'storage_type': config['type'],
          'endpoint': 'storage.googleapis.com',
          'region': 'US'}

encoded_image = db.sources.Files(**params)
frame = db.ops.ImageDecoder(img=encoded_image)

encoded_mask = db.sources.Files(**params)
mask_frame = db.ops.ImageDecoder(img=encoded_mask)

my_segment_imageset_class = db.ops.MySegment(
    frame=frame, mask=mask_frame,
    w=128, h=128,
    sigma1=1.0, sigma2=0.01,
    model_path=model_path)
output_op = db.sinks.FrameColumn(columns={'frame': my_segment_imageset_class})

job = Job(
    op_args={
        encoded_image: {'paths': image_files, **params},
        encoded_mask: {'paths': mask_files, **params},

        output_op: 'example_resized',
    })

start = time.time()
[out_table] = db.run(output_op, [job], force=True, work_packet_size=8, io_packet_size=32, tasks_in_queue_per_pu=4)
end = time.time()

print('Total time for instance segmentation in scanner: {0:.3f} sec'.format(end-start))
# out_table.column('frame').save_mp4(join(dataset, 'players', 'instance_segm.mp4'))
