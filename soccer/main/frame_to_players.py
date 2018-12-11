from scannerpy import  Database, Job
from soccer.main.kernels import *
import cv2
import glob
from os.path import join, basename
import os
import time
import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--cloud', action='store_true')
parser.add_argument('--bucket', default='', type=str)
parser.add_argument('--video', type=int, default=3, help='Margin around the pose')
parser.add_argument('--framestep', type=int, default=10, help='Margin around the pose')
parser.add_argument('--nworkers', type=int, default=0, help='Margin around the pose')
parser.add_argument('--work_packet_size', type=int, default=2, help='Margin around the pose')
parser.add_argument('--io_packet_size', type=int, default=4, help='Margin around the pose')


opt, _ = parser.parse_known_args()


path_to_data = opt.path_to_data

goal_dirs = [item for item in os.listdir(path_to_data) if os.path.isdir(os.path.join(path_to_data, item)) ]
goal_dirs.sort()

dataset = join(path_to_data,goal_dirs[opt.video])
print('Processing dataset: {0}'.format(dataset))

h, w = 1080, 1920

if opt.nworkers > 0:
    master = 'localhost:5001'
    workers = ['localhost:{:d}'.format(d) for d in range(5002, 5002 + opt.nworkers)]
    db = Database(master=master, workers=workers)
else:
    db = Database()

config = db.config.config['storage']
params = {'bucket': opt.bucket,
          'storage_type': config['type'],
          'endpoint': 'storage.googleapis.com',
          'region': 'US'}


# ======================================================================================================================
# Images and masks
image_files = glob.glob(join(dataset, 'images', '*.jpg'))
image_files.sort()

mask_files = glob.glob(join(dataset, 'detectron', '*.png'))
mask_files.sort()


# ======================================================================================================================
# Metadata
with open(join(dataset, 'metadata', 'poses.p'), 'rb') as f:
    openposes = pickle.load(f)

with open(join(dataset, 'metadata', 'calib.p'), 'rb') as f:
    calib_data = pickle.load(f)

frame_names = list(openposes.keys())
frame_names.sort()

pose_data = []
for fname in frame_names:
    n_poses = len(openposes[fname])
    poses_in_frame = []
    for i in range(n_poses):
        poses_in_frame.append(openposes[fname][i])
    data = {'poses': poses_in_frame,
            'A': calib_data[fname]['A'], 'R': calib_data[fname]['R'], 'T': calib_data[fname]['T']}
    pose_data.append(data)


# ======================================================================================================================
# Scanner calls
encoded_image = db.sources.Files(**params)
frame_img = db.ops.ImageDecoder(img=encoded_image)

encoded_mask = db.sources.Files(**params)
frame_mask = db.ops.ImageDecoder(img=encoded_mask)

data = db.sources.Python()
pass_data = db.ops.Pass(input=data)

draw_poses_class = db.ops.CropPlayersClass(image=frame_img, mask=frame_mask, metadata=pass_data, h=h, w=w, margin=0)
output_op = db.sinks.FrameColumn(columns={'frame': draw_poses_class})

job = Job(
    op_args={
        encoded_image: {'paths': image_files, **params},
        encoded_mask: {'paths': mask_files, **params},
        data: {'data': pickle.dumps(pose_data)},
        output_op: 'example_resized',
    })


start = time.time()
[out_table] = db.run(output_op, [job], force=True, work_packet_size=opt.work_packet_size,
                     io_packet_size=opt.io_packet_size, pipeline_instances_per_node=1, tasks_in_queue_per_pu=1)
end = time.time()
print('scanner pose drawing: {0:.4f} for {1} frames'.format(end-start, len(image_files)))

results = out_table.column('frame').load()


def mkdir(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)


mkdir(join(dataset, 'players'))
mkdir(join(dataset, 'players', 'images'))
mkdir(join(dataset, 'players', 'cnn_masks'))
mkdir(join(dataset, 'players', 'poseimgs'))

start = time.time()

for i, res in tqdm(enumerate(results)):
    buff = pickle.loads(res)

    for sel in range(len(buff)):
        _img = buff[sel]['img']
        _pose_img = buff[sel]['pose_img']
        _mask = buff[sel]['mask']

        framename = basename(image_files[i])[:-4]
        cv2.imwrite(join(dataset, 'players', 'images', '{0}_{1}.jpg'.format(framename, sel)), _img[:, :, ::-1])
        cv2.imwrite(join(dataset, 'players', 'cnn_masks', '{0}_{1}.png'.format(framename, sel)), _mask)
        cv2.imwrite(join(dataset, 'players', 'poseimgs', '{0}_{1}.png'.format(framename, sel)), _pose_img)
end = time.time()
print('Writing player files: {0:.4f} for {1} frames'.format(end-start, len(image_files)))



