from os.path import join
import glob
import argparse
import os

import time
from soccer.main.kernels import *
import matplotlib.pyplot as plt

import soccer.main.segment_op.build.segment_pb2 as segment_pb2

parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona/')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--cloud', action='store_true')
parser.add_argument('--bucket', default='', type=str)
opt, _ = parser.parse_known_args()


dataset = opt.path_to_data


db = Database()

config = db.config.config['storage']
params = {'bucket': opt.bucket,
          'storage_type': config['type'],
          'endpoint': 'storage.googleapis.com',
          'region': 'US'}


# ======================================================================================================================
# Images
image_files = glob.glob(join(dataset, 'images', '*.jpg'))
image_files.sort()

encoded_image = db.sources.Files(**params)
frame = db.ops.ImageDecoder(img=encoded_image)


# ======================================================================================================================
# Masks
mask_files = glob.glob(join(dataset, 'detectron', '*.png'))
mask_files.sort()

encoded_mask = db.sources.Files(**params)
mask_frame = db.ops.ImageDecoder(img=encoded_mask)


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

data = db.sources.Python()
pass_data = db.ops.Pass(input=data)


# ======================================================================================================================
# Scanner calls
draw_poses_class = db.ops.CropPlayersClass(image=frame, mask=mask_frame, metadata=pass_data, h=2160, w=3840, margin=0)
output_op = db.sinks.FrameColumn(columns={'frame': draw_poses_class})

job = Job(
    op_args={
        encoded_image: {'paths': image_files, **params},
        encoded_mask: {'paths': mask_files, **params},
        data: {'data': pickle.dumps(pose_data)},
        output_op: 'example_resized55',
    })

start = time.time()
[out_table] = db.run(output_op, [job], force=True)
results = out_table.column('frame').load()

end = time.time()
print('Total time for pose drawing in scanner: {0:.3f} sec'.format(end-start))


# ======================================================================================================================
# Instance Segmentation
# ======================================================================================================================


data = []
for i, res in enumerate(results):
    buff = pickle.loads(res)
    for sel in range(len(buff)):
        h, w = buff[sel]['img'].shape[:2]

        _img = segment_pb2.MyImage()
        _img.image_data = np.ndarray.tobytes(buff[sel]['img'].astype(np.uint8))

        _pose_img = segment_pb2.MyImage()
        _pose_img.image_data = np.ndarray.tobytes(buff[sel]['pose_img'].astype(np.uint8))

        data.append([_img.SerializeToString(), _pose_img.SerializeToString()])

        if i == 0:
            print(buff[sel]['img'].shape, buff[sel]['pose_img'].shape, len(_pose_img.image_data), len(_img.image_data), len(_img.SerializeToString()))
            break


db.new_table('test', ['img', 'pose_img'], data, force=True)


img = db.sources.FrameColumn()
pose_img = db.sources.FrameColumn()

cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.isfile(os.path.join(cwd, 'segment_op/build/libsegment_op.so')):
    print(
        'You need to build the custom op first: \n'
        '$ pushd {}/segment_op; mkdir build && cd build; cmake ..; make; popd'.
        format(cwd))
    exit()

# To load a custom op into the Scanner runtime, we use db.load_op to open the
# shared library we compiled. If the op takes arguments, it also optionally
# takes a path to the generated python file for the arg protobuf.
if opt.cloud:
    db.load_op(
        '/app/segment_op/build/libsegment_op.so',
        os.path.join(cwd, 'segment_op/build/segment_pb2.py'))
else:
    db.load_op(
        os.path.join(cwd, 'segment_op/build/libsegment_op.so'),
        os.path.join(cwd, 'segment_op/build/segment_pb2.py'))

model_path = 'model.yml.gz'

my_segment_imageset_class = db.ops.MySegment(
    frame=img, mask=pose_img,
    w=128, h=128,
    sigma1=1.0, sigma2=0.01,
    model_path=model_path)
output_op = db.sinks.FrameColumn(columns={'frame': my_segment_imageset_class})

job = Job(op_args={
    img: db.table('test').column('img'),
    pose_img: db.table('test').column('pose_img'),
    output_op: 'example_resized'
})


start = time.time()
[out_table] = db.run(output_op, [job], force=True)
end = time.time()
print('Total time for instance segmentation in scanner: {0:.3f} sec'.format(end-start))



















for i, res in enumerate(results):
    buff = pickle.loads(res)
    break

    for sel in range(len(buff)):
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(buff[sel]['img'])
        ax[1].imshow(buff[sel]['pose_img'])
        ax[2].imshow(buff[sel]['mask'])
        plt.show()
    break
# out_table.column('frame').save_mp4(join(dataset, 'players', 'poses'))
