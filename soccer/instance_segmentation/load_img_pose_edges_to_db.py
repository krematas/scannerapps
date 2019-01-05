from scannerpy import Database, Job
from os.path import join
import glob
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
    parser.add_argument('--table_name', default='data_for_inst_semg')

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
        poseimg_files = get_paths(join(dataset, 'players', 'poseimgs', '*.png'))
        edge_files = get_paths(join(dataset, 'players', 'edges', '*.png'))

    else:
        image_files = glob.glob(join(dataset, 'players', 'images', '*.jpg'))
        poseimg_files = glob.glob(join(dataset, 'players', 'poseimgs', '*.png'))
        edge_files = glob.glob(join(dataset, 'players', 'edges', '*.png'))

    image_files.sort()
    poseimg_files.sort()
    edge_files.sort()

    if opt.total_files > 0:
        image_files = image_files[:opt.total_files]
        poseimg_files = poseimg_files[:opt.total_files]
        edge_files = edge_files[:opt.total_files]

    print(len(image_files))

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

    encoded_image = db.sources.Files(**params)
    encoded_poseimg = db.sources.Files(**params)
    encoded_edges = db.sources.Files(**params)

    encoded_image2 = db.ops.Pass(input=encoded_image)
    encoded_poseimg2 = db.ops.Pass(input=encoded_poseimg)
    encoded_edges2 = db.ops.Pass(input=encoded_edges)

    output_op = db.sinks.Column(columns={'image': encoded_image2,
                                              'poseimg': encoded_poseimg2,
                                              'edges': encoded_edges2})

    job = Job(
        op_args={
            encoded_image: {'paths': image_files, **params},
            encoded_poseimg: {'paths': poseimg_files, **params},
            encoded_edges: {'paths': edge_files, **params},

            output_op: opt.table_name,
        })

    [_out_table] = db.run(output_op, [job], force=True, tasks_in_queue_per_pu=1)
    print(db.summarize())
