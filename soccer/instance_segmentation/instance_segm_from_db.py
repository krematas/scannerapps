import scannerpy
import cv2
import numpy as np
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType
from scannerpy.stdlib import pipelines
import time

from os.path import join, basename
import glob
import os
import subprocess as sp

import argparse
# import soccer.instance_segmentation.instancesegm_op.build.instancesegm_pb2 as instancesegm_pb2
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
    parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/emil-forsberg-goal-sweden-v-switzerland-match-55')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--cloud', action='store_true')
    parser.add_argument('--bucket', default='', type=str)
    parser.add_argument('--nworkers', type=int, default=0, help='Margin around the pose')
    parser.add_argument('--work_packet_size', type=int, default=2)
    parser.add_argument('--io_packet_size', type=int, default=4)
    parser.add_argument('--pipeline_instances_per_node', type=int, default=1)
    parser.add_argument('--tasks_in_queue_per_pu', type=int, default=1)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--table_name', default='data_for_inst_semg')

    opt, _ = parser.parse_known_args()

    if opt.pipeline_instances_per_node < 0:
        opt.pipeline_instances_per_node = None

    dataset = opt.path_to_data

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

    cwd = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isfile(os.path.join(cwd, 'instancesegm_op/build/libinstancesegm_op.so')):
        print(
            'You need to build the custom op first: \n'
            '$ pushd {}/instancesegm_op; mkdir build && cd build; cmake ..; make; popd'.format(cwd))
        exit()

    if opt.cloud:
        db.load_op(
            '/app/instancesegm_op/build/libinstancesegm_op.so',
            os.path.join(cwd, 'instancesegm_op/build/instancesegm_pb2.py'))
    else:
        db.load_op(
            os.path.join(cwd, 'instancesegm_op/build/libinstancesegm_op.so'),
            os.path.join(cwd, 'instancesegm_op/build/instancesegm_pb2.py'))

    config = db.config.config['storage']
    params = {'bucket': opt.bucket,
              'storage_type': config['type'],
              'endpoint': 'storage.googleapis.com',
              'region': 'US'}

    print(db.summarize())

    # @scannerpy.register_python_op(device_type=DeviceType.CPU)
    # def device_resize(config, frame: FrameType) -> FrameType:
    #     # plt.imshow(frame)
    #     # plt.show()
    #     return cv2.resize(frame, (config.args['width'], config.args['height']))
    #
    #
    # encoded_image2 = db.sources.Column()
    # frame2 = db.ops.ImageDecoder(img=encoded_image2)
    #
    # resized_frame = db.ops.device_resize(frame=frame2, width=128, height=128)
    # output = db.sinks.FrameColumn(columns={'frame': resized_frame})
    #
    # job = Job(op_args={
    #     encoded_image2: db.table(opt.table_name).column('image'),
    #     output: 'instance_segmentation'
    # })
    #
    # [table] = db.run(output=output, jobs=[job], force=True, work_packet_size=opt.work_packet_size,
    #                      io_packet_size=opt.io_packet_size, pipeline_instances_per_node=opt.pipeline_instances_per_node,
    #                      tasks_in_queue_per_pu=opt.tasks_in_queue_per_pu)
    #
    # # [table] = db.run(output=output, jobs=[job], force=True)
    #
    # import sys
    # sys.exit(-1)

    encoded_image = db.sources.Column()
    frame = db.ops.ImageDecoder(img=encoded_image)

    encoded_poseimg = db.sources.Column()
    poseimg_frame = db.ops.ImageDecoder(img=encoded_poseimg)

    encoded_edges = db.sources.Column()
    edge_frame = db.ops.ImageDecoder(img=encoded_edges)

    my_segment_imageset_class = db.ops.InstanceSegment(frame=frame, poseimg=poseimg_frame, edges=edge_frame,
                                                       sigma1=1.0, sigma2=0.01)
    output_op = db.sinks.FrameColumn(columns={'frame': my_segment_imageset_class})

    job = Job(
        op_args={
            encoded_image: db.table(opt.table_name).column('image'),
            encoded_poseimg: db.table(opt.table_name).column('poseimg'),
            encoded_edges: db.table(opt.table_name).column('edges'),

            output_op: 'instance_segmentation',
        })

    start = time.time()
    [out_table] = db.run(output_op, [job], force=True, work_packet_size=opt.work_packet_size,
                         io_packet_size=opt.io_packet_size, pipeline_instances_per_node=opt.pipeline_instances_per_node,
                         tasks_in_queue_per_pu=opt.tasks_in_queue_per_pu)
    end = time.time()
    print('Total time for instance segm in scanner: {0:.3f} sec'.format(end - start))

    tracename = join(dataset, 'instance.trace')
    if opt.cloud:
        tracename = 'instance-cloud.trace'
    out_table.profiler().write_trace(tracename)
    print('Trace saved in {0}'.format(tracename))

    if opt.save:

        def mkdir(path_to_dir):
            if not os.path.exists(path_to_dir):
                os.mkdir(path_to_dir)


        mkdir(join(dataset, 'players'))
        mkdir(join(dataset, 'players', 'masks'))

        results = out_table.column('frame').load()
        for i, res in enumerate(results):
            my_image = instancesegm_pb2.ProtoImage()
            my_image.ParseFromString(res)
            nparr = np.fromstring(my_image.image_data, np.uint8)
            instance_mask = nparr.reshape((my_image.h, my_image.w))

            framename = basename(image_files[i])[:-4]
            cv2.imwrite(join(dataset, 'players', 'masks', '{0}.png'.format(framename)), instance_mask)



    #
    # import matplotlib.pyplot as plt
    #
    # for i, res in enumerate(results):
    #     my_image = instancesegm_pb2.ProtoImage()
    #     my_image.ParseFromString(res)
    #     nparr = np.fromstring(my_image.image_data, np.uint8)
    #     instance_mask = nparr.reshape((my_image.h, my_image.w))
    #     plt.imshow(instance_mask)
    #     plt.show()
    #     print(i, my_image.w)
    # break


    # my_image = segment_pb2.MyImage()
