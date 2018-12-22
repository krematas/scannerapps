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

if __name__ == '__main__':

    # Testing settings
    parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
    parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/emil-forsberg-goal-sweden-v-switzerland-match-55')
    parser.add_argument('--path_to_model', default='/home/krematas/Mountpoints/grail/tmp/cnn/model.pth')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--cloud', action='store_true')
    parser.add_argument('--bucket', default='', type=str)
    parser.add_argument('--work_packet_size', type=int, default=2)
    parser.add_argument('--io_packet_size', type=int, default=4)

    opt, _ = parser.parse_known_args()


    @scannerpy.register_python_op(device_type=DeviceType.GPU, batch=20)
    class MyDepthEstimationClass(scannerpy.Kernel):
        def __init__(self, config):
            if opt.cloud:
               checkpoint = torch.load('model.pth')
            else:
                checkpoint = torch.load(config.args['model_path'])
            netG_state_dict = checkpoint['state_dict']
            netG = hg8(input_nc=4, output_nc=51)
            netG.load_state_dict(netG_state_dict)
            netG.cuda()

            self.logsoftmax = nn.LogSoftmax()
            self.normalize = transforms.Normalize(mean=[0.3402085, 0.42575407, 0.23771574],
                                             std=[0.1159472, 0.10461029, 0.13433486])

            self.img_size = config.args['img_size']
            self.net = netG
            pass

        def close(self):
            pass

        def execute(self, image: Sequence[FrameType], mask: Sequence[FrameType]) -> Sequence[FrameType]:

            batch_size = len(image)
            cur_batch = torch.zeros([batch_size, 4, self.img_size, self.img_size], dtype=torch.float32)

            for i in range(batch_size):
                _image = imresize(image[i], (self.img_size, self.img_size))
                _mask = imresize(mask[i][:, :, 0], (self.img_size, self.img_size), interp='nearest', mode='F')

                # ToTensor
                _image = _image.transpose((2, 0, 1)) / 255.0
                _mask = _mask[:, :, None].transpose((2, 0, 1)) / 255.0

                image_tensor = torch.from_numpy(_image)
                image_tensor = torch.FloatTensor(image_tensor.size()).copy_(image_tensor)
                image_tensor = self.normalize(image_tensor)
                mask_tensor = torch.from_numpy(_mask)

                cur_batch[i, :, :, :] = torch.cat((image_tensor, mask_tensor), 0)
            cur_batch = cur_batch.cuda()
            # Rescale
            # _image = imresize(image, (self.img_size, self.img_size))
            # _mask = imresize(mask[:, :, 0], (self.img_size, self.img_size), interp='nearest', mode='F')
            #
            # # ToTensor
            # _image = _image.transpose((2, 0, 1))/255.0
            # _mask = _mask[:, :, None].transpose((2, 0, 1))/255.0
            #
            # image_tensor = torch.from_numpy(_image)
            # image_tensor = torch.FloatTensor(image_tensor.size()).copy_(image_tensor)
            # mask_tensor = torch.from_numpy(_mask)
            #
            # # Normalize
            # image_tensor = self.normalize(image_tensor)
            #
            # # Make it BxCxHxW
            # image_tensor = image_tensor.unsqueeze(0)
            # mask_tensor = mask_tensor.unsqueeze(0)
            #
            # # Concat input and mask
            # image_tensor = torch.cat((image_tensor.float(), mask_tensor.float()), 1)
            # image_tensor = image_tensor.cuda()

            output = self.net(cur_batch)
            final_prediction = self.logsoftmax(output[-1])

            np_prediction = final_prediction.cpu().detach().numpy()
            np_prediction_list = [np_prediction[i, :, :, :].astype(np.float32) for i in range(batch_size)]

            return np_prediction_list

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

    my_depth_estimation_class = db.ops.MyDepthEstimationClass(image=frame, mask=mask_frame,
                                                              img_size=256, model_path=model_path, batch=20)
    output_op = db.sinks.FrameColumn(columns={'frame': my_depth_estimation_class})

    job = Job(
        op_args={
            encoded_image: {'paths': image_files, **params},
            encoded_mask: {'paths': mask_files, **params},
            output_op: 'example_resized5',
        })

    start = time.time()
    [out_table] = db.run(output_op, [job], force=True, work_packet_size=opt.work_packet_size,
                         io_packet_size=opt.io_packet_size)
    end = time.time()
    print('Total time for depth estimation in scanner: {0:.3f} sec'.format(end-start))

    # results = out_table.column('frame').load()
    #
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
