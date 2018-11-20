import scannerpy
import cv2
import numpy as np
import glob
from os.path import join, basename
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType
from scannerpy.stdlib import pipelines

import subprocess
import os.path
import numpy as np
import time
import utils as utils
import matplotlib.pyplot as plt

@scannerpy.register_python_op()
class CalibrationClass(scannerpy.Kernel):
    # __init__ is called once at the creation of the pipeline. Any arguments passed to the kernel
    # are provided through a protobuf object that you manually deserialize. See resize.proto for the
    # protobuf definition.
    def __init__(self, config):
        self.w = None # config.args['w']
        self.h = None # config.args['h']
        self.A = None # config.args['A']
        self.R = None # config.args['R']
        self.T = None # config.args['T']

    def new_stream(self, args):
        if args is None:
            return
        if 'w' in args:
            self.w = args['w']
        if 'h' in args:
            self.h = args['h']
        if 'A' in args:
            self.A = args['A']
        if 'R' in args:
            self.R = args['R']
        if 'T' in args:
            self.T = args['T']

    # execute is the core computation routine maps inputs to outputs, e.g. here resizes an input
    # frame to a smaller output frame.
    def execute(self, frame: FrameType, mask: FrameType) -> FrameType:
        edge_sfactor = 0.5
        edges = utils.robust_edge_detection(cv2.resize(frame[:, :, ::-1], None, fx=edge_sfactor, fy=edge_sfactor))
        edges = cv2.resize(edges, None, fx=1. / edge_sfactor, fy=1. / edge_sfactor)
        edges = cv2.Canny(edges.astype(np.uint8) * 255, 100, 200) / 255.0

        mask = cv2.dilate(mask[:, :, 0], np.ones((25, 25), dtype=np.uint8))

        edges = edges * (1 - mask)
        dist_transf = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 0)

        template, field_mask = utils.draw_field(self.A, self.R, self.T, self.h, self.w)

        II, JJ = (template > 0).nonzero()
        synth_field2d = np.array([[JJ, II]]).T[:, :, 0]

        # cv2.imwrite(time.asctime()+'.jpg', template*255)
        field3d = utils.plane_points_to_3d(synth_field2d, self.A, self.R, self.T)

        self.A, self.R, self.T = utils.calibrate_camera_dist_transf(self.A, self.R, self.T, dist_transf, field3d)

        rgb = frame.copy()
        canvas, mask = utils.draw_field(self.A, self.R, self.T, self.h, self.w)
        canvas = cv2.dilate(canvas.astype(np.uint8), np.ones((15, 15), dtype=np.uint8)).astype(float)
        rgb = rgb * (1 - canvas)[:, :, None] + np.dstack((canvas * 255, np.zeros_like(canvas), np.zeros_like(canvas)))

        # result = np.dstack((template, template, template))*255

        out = rgb.astype(np.uint8)
        # cv2.imwrite(time.asctime()+'.jpg', out)
        return out


path_to_data = '/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018'
dataset_list = [join(path_to_data, 'adnan-januzaj-goal-england-v-belgium-match-45'), join(path_to_data, 'ahmed-fathy-s-own-goal-russia-egypt')]
bucket = ''

db = Database()

config = db.config.config['storage']
params = {'bucket': bucket,
          'storage_type': config['type'],
          'endpoint': 'storage.googleapis.com',
          'region': 'US'}


video_list_scanner = []
video_names = []
imagename_list, maskname_list = [], []
calibs = []

for dataset in dataset_list:
    video_list_scanner.append((basename(dataset), join(dataset, 'video.mp4')))
    video_list_scanner.append((basename(dataset) + '_mask', join(dataset, 'mask.mp4')))

    video_names.append(basename(dataset))

    image_files = glob.glob(join(dataset, 'images', '*.jpg'))
    image_files.sort()
    imagename_list.append(image_files)

    mask_files = glob.glob(join(dataset, 'detectron', '*.png'))
    mask_files.sort()
    maskname_list.append(mask_files)

    cam_data = np.load(join(dataset, 'calib', '{0}.npy'.format(basename(image_files[0])[:-4]))).item()
    calibs.append(cam_data)


input_table, failed = db.ingest_videos(video_list_scanner, force=True)


frame = db.sources.FrameColumn()
mask = db.sources.FrameColumn()

encoded_image = db.sources.Files(**params)
frame_img = db.ops.ImageDecoder(img=encoded_image)

encoded_mask = db.sources.Files(**params)
frame_mask = db.ops.ImageDecoder(img=encoded_mask)


i = 1
calibrate_video_class = db.ops.CalibrationClass(frame=frame_img, mask=frame_mask)
output_op = db.sinks.FrameColumn(columns={'frame': calibrate_video_class})

jobs = []
for i in range(len(video_names)):

    job = Job(op_args={
        # frame: db.table(video_names[i]).column('frame'),
        # mask: db.table(video_names[i]+'_mask').column('frame'),

        encoded_image: {'paths': imagename_list[i], **params},
        encoded_mask: {'paths': maskname_list[i], **params},
        calibrate_video_class: {'w': 1920, 'h': 1080, 'A': calibs[i]['A'], 'R': calibs[i]['R'], 'T': calibs[i]['T']},
        output_op: 'example_resized_{0}'.format(i),
        })

    jobs.append(job)

start = time.time()
tables = db.run(output_op, jobs, force=True)
end = time.time()

for i in range(len(video_names)):
    savename = join(path_to_data, video_names[i], 'calib_scanner2')
    tables[i].column('frame').save_mp4(savename)
print('Successfully generated {0}_faces.mp4 in {1:.3f} secs'.format(savename, end-start))

# # ======================================================================================================================
# # Images
# # ======================================================================================================================
#
# total_files = -1
# image_files = glob.glob(join(dataset_list[0], 'images', '*.jpg'))
# image_files.sort()
# image_files = image_files[:total_files]
#
# encoded_image = db.sources.Files(**params)
# frame_img = db.ops.ImageDecoder(img=encoded_image)
#
# # ======================================================================================================================
# # Images
# # ======================================================================================================================
#
# mask_files = glob.glob(join(dataset_list[0], 'detectron', '*.png'))
# mask_files.sort()
# mask_files = mask_files[:total_files]
#
# encoded_mask = db.sources.Files(**params)
# frame_mask = db.ops.ImageDecoder(img=encoded_mask)
#
# basename = os.path.basename(image_files[0]).replace('.jpg', '')
# cam_data = np.load(join(dataset_list[0], 'calib', '{0}.npy'.format(basename))).item()
#
# # frame = db.sources.FrameColumn()
# # mask = db.sources.FrameColumn()
#
# calibrate_video_class = db.ops.CalibrationClass(frame=frame_img, mask=frame_mask, w=3840//2, h=2160//2, A=cam_data['A'], R=cam_data['R'], T=cam_data['T'])
# output_op = db.sinks.FrameColumn(columns={'frame': calibrate_video_class})
#
#
# jobs = []
# for dataset in dataset_list:
#     # ======================================================================================================================
#     # Images
#     # ======================================================================================================================
#
#     total_files = -1
#     image_files = glob.glob(join(dataset, 'images', '*.jpg'))
#     image_files.sort()
#     image_files = image_files[:total_files]
#
#     encoded_image = db.sources.Files(**params)
#     frame_img = db.ops.ImageDecoder(img=encoded_image)
#
#     # ======================================================================================================================
#     # Images
#     # ======================================================================================================================
#
#     mask_files = glob.glob(join(dataset, 'detectron', '*.png'))
#     mask_files.sort()
#     mask_files = mask_files[:total_files]
#
#     encoded_mask = db.sources.Files(**params)
#     frame_mask = db.ops.ImageDecoder(img=encoded_mask)
#
#     basename = os.path.basename(image_files[0]).replace('.jpg', '')
#     cam_data = np.load(join(dataset, 'calib', '{0}.npy'.format(basename))).item()
#
#     # frame = db.sources.FrameColumn()
#     # mask = db.sources.FrameColumn()
#
#     # calibrate_video_class = db.ops.CalibrationClass(frame=frame_img, mask=frame_mask, w=3840//2, h=2160//2, A=cam_data['A'], R=cam_data['R'], T=cam_data['T'])
#     # output_op = db.sinks.FrameColumn(columns={'frame': calibrate_video_class})
#
#     job = Job(op_args={
#         encoded_image: {'paths': image_files, **params},
#         encoded_mask: {'paths': mask_files, **params},
#         calibrate_video_class: {'w':3840//2, 'h':2160//2, 'A':cam_data['A'], 'R':cam_data['R'], 'T':cam_data['T']},
#         output_op: 'example_resized',
#     })
#     jobs.append(job)
#
#
# start = time.time()
# [out_table] = db.run(output_op, jobs, force=True)
# end = time.time()
#
# out_table.column('frame').save_mp4(join(dataset, 'calib_scanner.mp4'))
# print('Successfully generated {0}_faces.mp4 in {1:.3f} secs'.format(join(dataset, 'calib_scanner.mp4'), end-start))
