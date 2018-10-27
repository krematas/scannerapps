import scannerpy
import cv2
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType

from os.path import join
import numpy as np
import glob
import argparse
import pickle

import time
import cocoapi.PythonAPI.pycocotools.mask as mask_util

parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona/')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--cloud', action='store_true')
parser.add_argument('--bucket', default='', type=str)
opt, _ = parser.parse_known_args()


@scannerpy.register_python_op(name='DetectronInstSegm')
class DetectronInstSegm(scannerpy.Kernel):
    def __init__(self, config):
        self.w = config.args['w']
        self.h = config.args['h']

    def execute(self, image: FrameType, detectrondata: bytes) -> FrameType:

        detectrondata = pickle.loads(detectrondata)
        # boxes = detectrondata['boxes']
        # segms = detectrondata['segms']

        # areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # sorted_inds = np.argsort(-areas)

        instance_map = np.zeros((self.w, self.h))

        # for ii, i in enumerate(sorted_inds):
        #     masks = mask_util.decode(segms[i])
        #     instance_map += (masks * (ii + 100))

        return instance_map.astype(np.uint8)


@scannerpy.register_python_op()
class DrawPosesClass(scannerpy.Kernel):

    def __init__(self, config):
        self.w = config.args['w']
        self.h = config.args['h']
        self.limps = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 11], [11, 12], [12, 13], [1, 8],
             [8, 9], [9, 10], [14, 15], [16, 17], [0, 14], [0, 15], [14, 16], [15, 17]])

    def execute(self, image: FrameType, poses: bytes) -> FrameType:


        # output = np.zeros((self.h, self.w, 3), dtype=np.float32)
        output = image.copy()
        poses = pickle.loads(poses)
        poses = poses['data']
        for i in range(poses.shape[0]):
            keypoints = poses[i, :, :]

            lbl = i+200
            for k in range(self.limps.shape[0]):
                kp1, kp2 = self.limps[k, :].astype(int)
                bone_start = keypoints[kp1, :]
                bone_end = keypoints[kp2, :]
                bone_start[0] = np.maximum(np.minimum(bone_start[0], self.w - 1), 0.)
                bone_start[1] = np.maximum(np.minimum(bone_start[1], self.h - 1), 0.)

                bone_end[0] = np.maximum(np.minimum(bone_end[0], self.w - 1), 0.)
                bone_end[1] = np.maximum(np.minimum(bone_end[1], self.h - 1), 0.)

                if bone_start[2] > 0.0:
                    output[int(bone_start[1]), int(bone_start[0])] = 1
                    cv2.circle(output, (int(bone_start[0]), int(bone_start[1])), 2, (lbl, 0, 0), -1)

                if bone_end[2] > 0.0:
                    output[int(bone_end[1]), int(bone_end[0])] = 1
                    cv2.circle(output, (int(bone_end[0]), int(bone_end[1])), 2, (lbl, 0, 0), -1)

                if bone_start[2] > 0.0 and bone_end[2] > 0.0:
                    cv2.line(output, (int(bone_start[0]), int(bone_start[1])), (int(bone_end[0]), int(bone_end[1])),
                             (lbl, 0, 0), 1)

        return output.astype(np.uint8)


dataset = opt.path_to_data


image_files = glob.glob(join(dataset, 'images', '*.jpg'))
image_files.sort()


db = Database()

config = db.config.config['storage']
params = {'bucket': opt.bucket,
          'storage_type': config['type'],
          'endpoint': 'storage.googleapis.com',
          'region': 'US'}

encoded_image = db.sources.Files()
frame = db.ops.ImageDecoder(img=encoded_image)

with open(join(dataset, 'metadata', 'poses.p'), 'rb') as f:
    openposes = pickle.load(f)

frame_names = list(openposes.keys())
frame_names.sort()
#
# pose_data = []
# for fname in frame_names:
#     n_poses = len(openposes[fname])
#     poses_in_frame = np.zeros((n_poses, 18, 3), dtype=np.float32)
#     for i in range(n_poses):
#         poses_in_frame[i, :, :] = openposes[fname][i]
#     data = {'n_poses': n_poses, 'data': poses_in_frame}
#     pose_data.append(data)
#
# data = db.sources.Python()
# pass_data = db.ops.Pass(input=data)
#
# draw_poses_class = db.ops.DrawPosesClass(image=frame, poses=pass_data, h=2160, w=3840)
# output_op = db.sinks.FrameColumn(columns={'frame': draw_poses_class})
#
# job = Job(
#     op_args={
#         encoded_image: {'paths': image_files, **params},
#         data: {'data': pickle.dumps(pose_data)},
#         output_op: 'example_resized55',
#     })
#
# start = time.time()
# [out_table] = db.run(output_op, [job], force=True)
# end = time.time()
#
#
# print('Total time for pose drawing in scanner: {0:.3f} sec'.format(end-start))
#
#
# out_table.column('frame').save_mp4(join(dataset, 'players', 'poses'))

with open(join(dataset, 'metadata', 'detectron.p'), 'rb') as f:
    detectron = pickle.load(f)

detectron_data = []
for fname in frame_names:
    detectron_data.append({'boxes': detectron[fname]['boxes']})

detectrondata = db.sources.Python()
pass_detectron = db.ops.Pass(input=detectrondata)

draw_detectron_class = db.ops.DetectronInstSegm(image=frame, detectrondata=pass_detectron, w=10, h=10)
output_op = db.sinks.FrameColumn(columns={'frame': draw_detectron_class})

print(len(pickle.dumps(detectron_data)))

job = Job(
    op_args={
        encoded_image: {'paths': image_files, **params},
        detectrondata: {'data': pickle.dumps(detectron_data)},
        output_op: 'example_resized55',
    })

start = time.time()
[out_table] = db.run(output_op, [job], force=True)
end = time.time()

print('Total time for instance drawing in scanner: {0:.3f} sec'.format(end-start))
# out_table.column('frame').save_mp4(join(dataset, 'players', 'segms'))