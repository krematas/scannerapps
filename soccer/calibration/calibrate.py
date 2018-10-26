import scannerpy
import cv2
import numpy as np
import utils as utils
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType
from scannerpy.stdlib import pipelines

import subprocess
import os.path
import numpy as np

# A kernel file defines a standalone Python kernel which performs some computation by exporting a
# Kernel class.
@scannerpy.register_python_op()
class CalibrationClass(scannerpy.Kernel):
    # __init__ is called once at the creation of the pipeline. Any arguments passed to the kernel
    # are provided through a protobuf object that you manually deserialize. See resize.proto for the
    # protobuf definition.
    def __init__(self, config):
        self._w = config.args['w']
        self._h = config.args['h']
        self._A = config.args['A']
        self._R = config.args['R']
        self._T = config.args['T']

    # execute is the core computation routine maps inputs to outputs, e.g. here resizes an input
    # frame to a smaller output frame.
    def execute(self, frame: FrameType, mask: FrameType) -> FrameType:
        edge_sfactor = 0.5
        edges = utils.robust_edge_detection(cv2.resize(frame, None, fx=edge_sfactor, fy=edge_sfactor))
        edges = cv2.resize(edges, None, fx=1. / edge_sfactor, fy=1. / edge_sfactor)
        edges = cv2.Canny(edges.astype(np.uint8) * 255, 100, 200) / 255.0

        mask = cv2.dilate(mask[:, :, 0], np.ones((25, 25), dtype=np.uint8))

        edges = edges * (1 - mask)
        dist_transf = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 0)

        template, field_mask = utils.draw_field(self._A, self._R, self._T, self._h, self._w)

        II, JJ = (template > 0).nonzero()
        synth_field2d = np.array([[JJ, II]]).T[:, :, 0]

        field3d = utils.plane_points_to_3d(synth_field2d, self._A, self._R, self._T)

        self._A, self._R, self._T = utils.calibrate_camera_dist_transf(self._A, self._R, self._T, dist_transf, field3d)

        rgb = frame.copy()
        canvas, mask = utils.draw_field(self._A, self._R, self._T, self._h, self._w)
        canvas = cv2.dilate(canvas.astype(np.uint8), np.ones((15, 15), dtype=np.uint8)).astype(float)
        rgb = rgb * (1 - canvas)[:, :, None] + np.dstack((canvas * 255, np.zeros_like(canvas), np.zeros_like(canvas)))

        # result = np.dstack((template, template, template))*255

        out = rgb.astype(np.uint8)
        return out



movie_path = '/home/krematas/Mountpoints/grail/data/barcelona/test.mp4'
print('Detecting faces in movie {}'.format(movie_path))
movie_name = os.path.splitext(os.path.basename(movie_path))[0]

mask_path = '/home/krematas/Mountpoints/grail/data/barcelona/mask.mp4'
db = Database()
print('Ingesting video into Scanner ...')
input_tables, failed = db.ingest_videos([(movie_name, movie_path), ('mask', mask_path)], force=True)

print(db.summarize())
print('Failures:', failed)

cam_data = np.load('/home/krematas/Mountpoints/grail/data/barcelona/calib/00114.npy').item()


# db.register_op('Calibrate', [('frame', ColumnType.Video), ('mask', ColumnType.Video)], [('resized', ColumnType.Video)])

# Custom Python kernels for ops reside in a separate file, here calibrate_kernel.py.
# cwd = '/home/krematas/code/scanner/examples/apps/soccer'
# db.register_python_kernel('Calibrate', DeviceType.CPU, cwd + '/calibrate_kernel.py')

frame = db.sources.FrameColumn()
mask = db.sources.FrameColumn()

# Then we use our op just like in the other examples.
calibrate_video_class = db.ops.CalibrationClass(frame=frame, mask=mask, w=3840, h=2160, A=cam_data['A'], R=cam_data['R'], T=cam_data['T'])
output_op = db.sinks.FrameColumn(columns={'frame': calibrate_video_class})

job = Job(op_args={
    frame: input_tables[0].column('frame'),
    mask: input_tables[1].column('frame'),
    output_op: 'example_resized',
})
[out_table] = db.run(output_op, [job], force=True)
out_table.column('frame').save_mp4(movie_name + '_faces')

print('Successfully generated {:s}_faces.mp4'.format(movie_name))
