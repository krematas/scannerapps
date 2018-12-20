import scannerpy
import numpy as np
import os
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType
from typing import Sequence

from os.path import join
import glob

import torch
import torch.nn as nn
from torchvision import transforms

from soccer.depth.hourglass import hg8
import matplotlib.pyplot as plt
from scipy.misc import imresize

import argparse
import time
import subprocess as sp

if __name__ == '__main__':

    # Testing settings
    parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
    parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/emil-forsberg-goal-sweden-v-switzerland-match-55')
    parser.add_argument('--path_to_model', default='/home/krematas/Mountpoints/grail/tmp/cnn/model.pth')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--cloud', action='store_true')
    parser.add_argument('--bucket', default='', type=str)

    opt, _ = parser.parse_known_args()


    @scannerpy.register_python_op(batch=50)
    class TestPyBatch(scannerpy.Kernel):
        def __init__(self, config):
            self.img_size = config.args['img_size']
            pass

        def close(self):
            pass

        def execute(self, frame: Sequence[FrameType], mask: Sequence[FrameType]) -> Sequence[FrameType]:
            batch_size = len(frame)
            print(len(frame), len(mask), frame[0].shape, mask[0].shape)
            # out = []
            # for i in range(batch_size):
            #     image = imresize(frame[i], (self.img_size, self.img_size))
            #     mask = imresize(mask[i], (self.img_size, self.img_size), interp='nearest', mode='F')
            #     out.append(image*mask)
            return [imresize(frame[i], (self.img_size, self.img_size)) for i in range(batch_size)]


    dataset = opt.path_to_data

    if opt.cloud:
        def get_paths(path):
            paths = sp.check_output('gsutil ls gs://{:s}/{:s}'.format(opt.bucket, path),
                                    shell=True).strip().decode('utf-8')
            paths = paths.split('\n')
            prefix_len = len('gs://{:s}'.format(opt.bucket))
            stripped_paths = [p[prefix_len:] for p in paths]
            return stripped_paths
        image_files = get_paths(join(dataset, 'players', 'images', '*.jpg'))
        mask_files = get_paths(join(dataset, 'players', 'masks', '*.png'))
    else:
        image_files = glob.glob(join(dataset, 'players', 'images', '*.jpg'))
        mask_files = glob.glob(join(dataset, 'players', 'masks', '*.png'))

    image_files.sort()
    mask_files.sort()

    model_path = opt.path_to_model

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

    config = db.config.config['storage']
    params = {'bucket': opt.bucket,
              'storage_type': config['type'],
              'endpoint': 'storage.googleapis.com',
              'region': 'US'}

    encoded_image = db.sources.Files(**params)
    frame = db.ops.ImageDecoder(img=encoded_image)

    encoded_mask = db.sources.Files(**params)
    mask_frame = db.ops.ImageDecoder(img=encoded_mask)

    my_depth_estimation_class = db.ops.TestPyBatch(frame=frame, mask=mask_frame, img_size=256, batch=50)
    output_op = db.sinks.FrameColumn(columns={'frame': my_depth_estimation_class})

    job = Job(
        op_args={
            encoded_image: {'paths': image_files, **params},
            encoded_mask: {'paths': mask_files, **params},
            output_op: 'example_resized5',
        })

    start = time.time()
    [out_table] = db.run(output_op, [job], force=True, work_packet_size=16, io_packet_size=16)
    end = time.time()
    print('Total time for depth estimation in scanner: {0:.3f} sec'.format(end-start))

    results = out_table.column('frame').load()

    # path_to_save = join(dataset, 'players', 'prediction_scanner')
    # if not os.path.exists(path_to_save):
    #     os.mkdir(path_to_save)
    #
    # for i, res in enumerate(results):
    #     pred_scanner = np.argmax(res, axis=0)
    #     np.save(join(path_to_save, '{0:05d}.npy'.format(i)), res)

        # if opt.visualize:
        #     # Visualization
        #     pred = np.load(pred_files[i])[0, :, :, :]
        #     pred = np.argmax(pred, axis=0)
        #     fig, ax = plt.subplots(1, 2)
        #
        #     ax[1].imshow(pred)
        #     ax[0].imshow(pred_scanner)
        #     plt.show()
