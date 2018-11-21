import cv2
import numpy as np
import glob
from os.path import join, basename

import soccer.calibration.utils as utils
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis
import time

path_to_data = '/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018'
dataset_list = [join(path_to_data, 'adnan-januzaj-goal-england-v-belgium-match-45'), join(path_to_data, 'ahmed-fathy-s-own-goal-russia-egypt'), join(path_to_data, 'ahmed-musa-1st-goal-nigeria-iceland'), join(path_to_data, 'ahmed-musa-2nd-goal-nigeria-iceland')]

dataset = join(path_to_data, 'ahmed-fathy-s-own-goal-russia-egypt')
image_files = glob.glob(join(dataset, 'images', '*.jpg'))
image_files.sort()

mask_files = glob.glob(join(dataset, 'detectron', '*.png'))
mask_files.sort()

cam_data = np.load(join(dataset, 'calib', '{0}.npy'.format(basename(image_files[0])[:-4]))).item()

h, w = 1080, 1920
A, R, T = cam_data['A'],  cam_data['R'],  cam_data['T']


for i in range(len(image_files)):

    print('{0} ========================================================='.format(i))

    frame = cv2.imread(image_files[i])
    mask = cv2.imread(mask_files[i])
    edge_sfactor = 1.0
    edges = utils.robust_edge_detection(cv2.resize(frame[:, :, ::-1], None, fx=edge_sfactor, fy=edge_sfactor))
    mask = cv2.dilate(mask[:, :, 0], np.ones((25, 25), dtype=np.uint8))/255

    edges = edges * (1 - mask)

    start = time.time()
    skel = medial_axis(edges, return_distance=False)
    end = time.time()
    print('skimage: {0:.4f}'.format(end-start))


    start = time.time()
    dist_transf = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 0)
    end = time.time()
    print('distance transform: {0:.4f}'.format(end-start))

    start = time.time()
    template, field_mask = utils.draw_field(A, R, T, h, w)
    end = time.time()
    print('draw: {0:.4f}'.format(end-start))

    II, JJ = (template > 0).nonzero()
    synth_field2d = np.array([[JJ, II]]).T[:, :, 0]

    # cv2.imwrite(time.asctime()+'.jpg', template*255)
    field3d = utils.plane_points_to_3d(synth_field2d, A, R, T)

    start = time.time()
    A, R, T = utils.calibrate_camera_dist_transf(A, R, T, dist_transf, field3d)
    end = time.time()
    print('optim: {0:.4f}\n\n'.format(end-start))

    rgb = frame.copy()
    canvas, mask = utils.draw_field(A, R, T, h, w)
    canvas = cv2.dilate(canvas.astype(np.uint8), np.ones((15, 15), dtype=np.uint8)).astype(float)
    rgb = rgb * (1 - canvas)[:, :, None] + np.dstack((canvas * 255, np.zeros_like(canvas), np.zeros_like(canvas)))

    # result = np.dstack((template, template, template))*255

    out = rgb.astype(np.uint8)

    if i % 25 == 0:
        plt.imshow(out)
        plt.show()


