import scannerpy
import cv2
import numpy as np
import glob
from os.path import join, basename
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType
from scannerpy.stdlib import pipelines
from skimage.morphology import medial_axis

import subprocess
import os.path
import numpy as np
import time
import utils as utils
import matplotlib.pyplot as plt
import pickle

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


path_to_data = '/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018'
dataset_list = [join(path_to_data, 'adnan-januzaj-goal-england-v-belgium-match-45'), join(path_to_data, 'ahmed-fathy-s-own-goal-russia-egypt'), join(path_to_data, 'ahmed-musa-1st-goal-nigeria-iceland'), join(path_to_data, 'ahmed-musa-2nd-goal-nigeria-iceland')]
dataset_list = [join(path_to_data, 'ahmed-musa-1st-goal-nigeria-iceland')]

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


encoded_image = db.sources.Files(**params)
frame_img = db.ops.ImageDecoder(img=encoded_image)

encoded_mask = db.sources.Files(**params)
frame_mask = db.ops.ImageDecoder(img=encoded_mask)


i = 0
dist_transform_class = db.ops.DistanceTransformClass(frame=frame_img, mask=frame_mask)
output_op = db.sinks.FrameColumn(columns={'frame': dist_transform_class})

job = Job(
    op_args={
    encoded_image: {'paths': imagename_list[i], **params},
    encoded_mask: {'paths': maskname_list[i], **params},
    output_op: 'example_resized55',
})


start = time.time()
[out_table] = db.run(output_op, [job], force=True)
end = time.time()
print('scanner distance transform: {0:.4f}'.format(end-start))