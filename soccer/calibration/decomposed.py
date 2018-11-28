import scannerpy
import cv2
import glob
from os.path import join, basename
from scannerpy import Database, Job, FrameType
from skimage.morphology import medial_axis
import os

import numpy as np
import time
import soccer.calibration.utils as utils
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--cloud', action='store_true')
parser.add_argument('--bucket', default='', type=str)
parser.add_argument('--nframes', type=int, default=5, help='Margin around the pose')
parser.add_argument('--framestep', type=int, default=10, help='Margin around the pose')
parser.add_argument('--nworkers', type=int, default=5, help='Margin around the pose')
parser.add_argument('--work_packet_size', type=int, default=4, help='Margin around the pose')
parser.add_argument('--io_packet_size', type=int, default=8, help='Margin around the pose')


opt, _ = parser.parse_known_args()


@scannerpy.register_python_op()
class DistanceTransformClass(scannerpy.Kernel):
    # __init__ is called once at the creation of the pipeline. Any arguments passed to the kernel
    # are provided through a protobuf object that you manually deserialize. See resize.proto for the
    # protobuf definition.
    def __init__(self, config):
        self.edge_sfactor = 1.0

    # execute is the core computation routine maps inputs to outputs, e.g. here resizes an input
    # frame to a smaller output frame.
    def execute(self, frame: FrameType, mask: FrameType) -> bytes:
        edges = utils.robust_edge_detection(cv2.resize(frame[:, :, ::-1], None,
                                                       fx=self.edge_sfactor, fy=self.edge_sfactor))
        skel = medial_axis(edges, return_distance=False)
        edges = skel.astype(np.uint8)

        mask = cv2.dilate(mask[:, :, 0], np.ones((25, 25), dtype=np.uint8)) / 255

        edges = edges * (1 - mask)
        dist_transf = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 0)

        return pickle.dumps(dist_transf)


path_to_data = opt.path_to_data

goal_dirs = [item for item in os.listdir(path_to_data) if os.path.isdir(os.path.join(path_to_data, item)) ]
goal_dirs.sort()

goal_id = 41
dataset = join(path_to_data,goal_dirs[goal_id])


master = 'localhost:5001'
workers = ['localhost:{:d}'.format(d) for d in range(5002, 5002+opt.nworkers)]
# db = Database(master=master, workers=workers)
db = Database()

config = db.config.config['storage']
params = {'bucket': opt.bucket,
          'storage_type': config['type'],
          'endpoint': 'storage.googleapis.com',
          'region': 'US'}

image_files = glob.glob(join(dataset, 'images', '*.jpg'))
image_files.sort()

mask_files = glob.glob(join(dataset, 'detectron', '*.png'))
mask_files.sort()

cam_data = np.load(join(dataset, 'calib', '{0}.npy'.format(basename(image_files[0])[:-4]))).item()


image_files = [image_files[i] for i in range(0, len(image_files), opt.framestep)]
mask_files = [mask_files[i] for i in range(0, len(mask_files), opt.framestep)]


encoded_image = db.sources.Files(**params)
frame_img = db.ops.ImageDecoder(img=encoded_image)

encoded_mask = db.sources.Files(**params)
frame_mask = db.ops.ImageDecoder(img=encoded_mask)


dist_transform_class = db.ops.DistanceTransformClass(frame=frame_img, mask=frame_mask)
output_op = db.sinks.FrameColumn(columns={'frame': dist_transform_class})

job = Job(
    op_args={
    encoded_image: {'paths': image_files, **params},
    encoded_mask: {'paths': mask_files, **params},
    output_op: 'example_resized55',
})


start = time.time()
[out_table] = db.run(output_op, [job], force=True, pipeline_instances_per_node=1,
                     work_packet_size=opt.work_packet_size, io_packet_size=opt.io_packet_size)
end = time.time()
print('scanner distance transform: {0:.4f}'.format(end-start))


# ======================================================================================================================
results = out_table.column('frame').load()
# dist_transf_pickles = [res for res in results]
dist_transf_list = [pickle.loads(res) for res in results]

A, R, T = cam_data['A'], cam_data['R'], cam_data['T']
h, w = 1080, 1920

n_frames = len(image_files)
start = time.time()
for j in tqdm(range(n_frames)):
    dist_transf = dist_transf_list[j]
    start = time.time()
    template, field_mask = utils.draw_field(A, R, T, h, w)
    end = time.time()
    # print('draw: {0:.4f}'.format(end-start))

    II, JJ = (template > 0).nonzero()
    synth_field2d = np.array([[JJ, II]]).T[:, :, 0]
    field3d = utils.plane_points_to_3d(synth_field2d, A, R, T)

    start = time.time()
    A, R, T = utils.calibrate_camera_dist_transf(A, R, T, dist_transf, field3d)
    end = time.time()
    # print('optim: {0:.4f}\n\n'.format(end - start))

    if j == n_frames-1:
        frame = cv2.imread(image_files[j])[:, :, ::-1]
        rgb = frame.copy()
        canvas, mask = utils.draw_field(A, R, T, h, w)
        canvas = cv2.dilate(canvas.astype(np.uint8), np.ones((15, 15), dtype=np.uint8)).astype(float)
        rgb = rgb * (1 - canvas)[:, :, None] + np.dstack((canvas * 255, np.zeros_like(canvas), np.zeros_like(canvas)))

        # result = np.dstack((template, template, template))*255

        out = rgb.astype(np.uint8)

        end = time.time()
        print('calibration: {0:.4f}'.format(end - start))

        plt.imshow(out)
        plt.show()
    #
    # if j == 10:
    #     break



