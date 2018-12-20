import scannerpy
import cv2
from scannerpy import Database, DeviceType, Job, FrameType
from os.path import join
import glob
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

    db = Database()

    config = db.config.config['storage']
    params = {'bucket': opt.bucket,
              'storage_type': config['type'],
              'endpoint': 'storage.googleapis.com',
              'region': 'US'}

    image_files = glob.glob(join(dataset, 'players', 'images', '*.jpg'))
    image_files.sort()

    if opt.total_files > 0:
        image_files = image_files[:opt.total_files]


    @scannerpy.register_python_op(device_type=DeviceType.CPU)
    def device_resize(config, frame: FrameType) -> FrameType:
        return cv2.resize(frame, (config.args['width'], config.args['height']))


    encoded_image = db.sources.Files(**params)
    encoded_poseimg = db.sources.Files(**params)
    encoded_edges = db.sources.Files(**params)

    output_op = db.sinks.FrameColumn(columns={'image': encoded_image})

    job = Job(
        op_args={
            encoded_image: {'paths': image_files, **params},
            output_op: opt.table_name,
        })

    [_out_table] = db.run(output_op, [job], force=True, tasks_in_queue_per_pu=1)

    print(db.summarize())

    encoded_image2 = db.sources.FrameColumn()
    frame2 = db.ops.ImageDecoder(img=encoded_image2)

    resized_frame = db.ops.device_resize(frame=frame2, width=128, height=128)
    output = db.sinks.FrameColumn(columns={'frame': resized_frame})

    job = Job(op_args={
        encoded_image2: db.table(opt.table_name).column('image'),
        output: 'instance_segmentation'
    })

    [table] = db.run(output=output, jobs=[job], force=True, work_packet_size=opt.work_packet_size,
                     io_packet_size=opt.io_packet_size, pipeline_instances_per_node=opt.pipeline_instances_per_node,
                     tasks_in_queue_per_pu=opt.tasks_in_queue_per_pu)