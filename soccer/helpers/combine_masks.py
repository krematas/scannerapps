import scannerpy
import cv2
import numpy as np
from scannerpy import Database, DeviceType, Job, FrameType
import time
import pickle
from os.path import join, basename
import glob
import os
import subprocess as sp
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
    parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/emil-forsberg-goal-sweden-v-switzerland-match-55')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--cloud', action='store_true')
    parser.add_argument('--bucket', default='', type=str)
    parser.add_argument('--nworkers', type=int, default=0, help='Margin around the pose')
    parser.add_argument('--total_files', type=int, default=-1)
    parser.add_argument('--work_packet_size', type=int, default=2)
    parser.add_argument('--io_packet_size', type=int, default=4)
    parser.add_argument('--pipeline_instances_per_node', type=int, default=1)
    parser.add_argument('--tasks_in_queue_per_pu', type=int, default=1)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--table_name', default='instance_data')

    opt, _ = parser.parse_known_args()

    if opt.pipeline_instances_per_node < 0:
        opt.pipeline_instances_per_node = None

    dataset = opt.path_to_data

    if opt.cloud:

        def get_paths(path):
            paths = sp.check_output('gsutil ls gs://{:s}/{:s}'.format(opt.bucket, path),
                                    shell=True).strip().decode('utf-8')
            paths = paths.split('\n')
            prefix_len = len('gs://{:s}/'.format(opt.bucket))
            stripped_paths = [p[prefix_len:] for p in paths]
            return stripped_paths


        cnn_files = get_paths(join(dataset, 'players', 'cnn_masks', '*.png'))
        inst_files = get_paths(join(dataset, 'players', 'instance_masks', '*.png'))

    else:
        cnn_files = glob.glob(join(dataset, 'players', 'cnn_masks', '*.png'))
        inst_files = glob.glob(join(dataset, 'players', 'instance_masks', '*.png'))

    cnn_files.sort()
    inst_files.sort()

    if opt.total_files > 0:
        cnn_files = cnn_files[:opt.total_files]
        inst_files = inst_files[:opt.total_files]

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
        db = Database(master=master, start_cluster=False, config_path='./config.toml', grpc_timeout=60)
        print('db was created.')
    else:
        db = Database()

    config = db.config.config['storage']
    params = {'bucket': opt.bucket,
              'storage_type': config['type'],
              'endpoint': 'storage.googleapis.com',
              'region': 'US'}

    encoded_cnn_files = db.sources.Files(**params)
    encoded_inst_files = db.sources.Files(**params)

    cnn_frame = db.ops.ImageDecoder(img=encoded_cnn_files)
    inst_frame = db.ops.ImageDecoder(img=encoded_inst_files)


    @scannerpy.register_python_op(device_type=DeviceType.CPU)
    def device_combine_masks(config, cnn_mask: FrameType, inst_mask: FrameType) -> bytes:
        out = cnn_mask*inst_mask
        return pickle.dumps(out)


    combined_mask = db.ops.device_combine_masks(cnn_mask=cnn_frame, inst_mask=inst_frame)
    output_op = db.sinks.FrameColumn(columns={'frame': combined_mask})

    job = Job(
        op_args={
            encoded_cnn_files: {'paths': cnn_files, **params},
            encoded_inst_files: {'paths': inst_files, **params},

            output_op: 'masks',
        })

    start = time.time()
    [out_table] = db.run(output_op, [job], force=True, work_packet_size=opt.work_packet_size,
                         io_packet_size=opt.io_packet_size, pipeline_instances_per_node=opt.pipeline_instances_per_node,
                         tasks_in_queue_per_pu=opt.tasks_in_queue_per_pu)
    end = time.time()
    print('Total time for instance segm in scanner: {0:.3f} sec for {1} images'.format(end - start, len(cnn_files)))

    if opt.save:

        def mkdir(path_to_dir):
            if not os.path.exists(path_to_dir):
                os.mkdir(path_to_dir)

        mkdir(join(dataset, 'players'))
        mkdir(join(dataset, 'players', 'masks'))

        start = time.time()
        results = out_table.column('frame').load()
        for i, res in enumerate(results):

            buff = pickle.loads(res)
            if i == 0:
                print(buff.shape)
            framename = basename(cnn_files[i])[:-4]
            cv2.imwrite(join(dataset, 'players', 'masks', '{0}.png'.format(framename)), buff*255)
        end = time.time()
        print('Files saved in {0:.3f} secs'.format(end-start))