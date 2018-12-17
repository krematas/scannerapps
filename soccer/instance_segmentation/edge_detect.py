import cv2
import numpy as np
from scannerpy import Database, Job
import edges_op.build.edges_pb2 as edges_pb2
import time
from os.path import join, basename
import glob
import os
import subprocess as sp
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
    parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona/')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--cloud', action='store_true')
    parser.add_argument('--bucket', default='', type=str)
    parser.add_argument('--nworkers', type=int, default=0, help='Margin around the pose')
    parser.add_argument('--total_files', type=int, default=-1)
    parser.add_argument('--work_packet_size', type=int, default=2)
    parser.add_argument('--io_packet_size', type=int, default=4)
    parser.add_argument('--pipeline_instances_per_node', type=int, default=1)
    parser.add_argument('--save', action='store_true')

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


        image_files = get_paths(join(dataset, 'players', 'images', '*.jpg'))
    else:
        image_files = glob.glob(join(dataset, 'players', 'images', '*.jpg'))

    image_files.sort()
    if opt.total_files > 0:
        image_files = image_files[:opt.total_files]
    print('Total files: {0}'.format(len(image_files)))

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
    # cwd = '/home/krematas/code/scannerapps/soccer/instance_segmentation/'
    if not os.path.isfile(os.path.join(cwd, 'edges_op/build/libedges_op.so')):
        print(
            'You need to build the custom op first: \n'
            '$ pushd {}/edges_op; mkdir build && cd build; cmake ..; make; popd'.format(cwd))
        exit()

    # To load a custom op into the Scanner runtime, we use db.load_op to open the
    # shared library we compiled. If the op takes arguments, it also optionally
    # takes a path to the generated python file for the arg protobuf.
    if opt.cloud:
        db.load_op('/app/edges_op/build/libedges_op.so', os.path.join(cwd, 'edges_op/build/edges_pb2.py'))
    else:
        db.load_op(
            os.path.join(cwd, 'edges_op/build/libedges_op.so'),
            os.path.join(cwd, 'edges_op/build/edges_pb2.py'))

    config = db.config.config['storage']
    params = {'bucket': opt.bucket,
              'storage_type': config['type'],
              'endpoint': 'storage.googleapis.com',
              'region': 'US'}

    encoded_image = db.sources.Files(**params)
    frame = db.ops.ImageDecoder(img=encoded_image)

    my_edge_detection_class = db.ops.EdgeDetection(frame=frame, model_path='model.yml.gz')
    output_op = db.sinks.FrameColumn(columns={'frame': my_edge_detection_class})

    job = Job(
        op_args={
            encoded_image: {'paths': image_files, **params},
            output_op: 'example_resized',
        })

    start = time.time()
    [out_table] = db.run(output_op, [job], force=True, work_packet_size=opt.work_packet_size,
                         io_packet_size=opt.io_packet_size, pipeline_instances_per_node=opt.pipeline_instances_per_node,
                         tasks_in_queue_per_pu=1)
    end = time.time()
    print('Total time for edge detection in scanner: {0:.3f} sec for {1} images'.format(end - start, len(image_files)))

    tracename = 'edge.trace'
    if opt.cloud:
        tracename = 'edge-cloud.trace'
    out_table.profiler().write_trace(join(dataset, tracename))
    print('Trace saved in {0}'.format(join(dataset, tracename)))

    if opt.save:
        results = out_table.column('frame').load()

        def mkdir(path_to_dir):
            if not os.path.exists(path_to_dir):
                os.mkdir(path_to_dir)

        mkdir(join(dataset, 'players', 'edges'))

        for i, res in enumerate(results):
            my_image = edges_pb2.ProtoImage()
            my_image.ParseFromString(res)
            nparr = np.fromstring(my_image.image_data, np.float32)
            edges = nparr.reshape((my_image.h, my_image.w))

            framename = basename(image_files[i])[:-4]
            cv2.imwrite(join(dataset, 'players', 'edges', '{0}.png'.format(framename)), edges*255)
