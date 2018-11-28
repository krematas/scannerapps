import torch
import soccer.calibration.utils as utils
import numpy as np
from os.path import join
import torch.nn.functional as F
import cv2
from skimage.morphology import medial_axis
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import interpolate
import time


data_path = '/home/krematas/Mountpoints/grail/data/Singleview/Soccer/'
path_to_field3d = join(data_path, 'field.ply')

vertex_data, _, _, _ = utils.read_ply(path_to_field3d)
vertices, _, _ = utils.ply_to_numpy(vertex_data)
n_vertices = vertices.shape[0]

dataset = 'Russia2018/adnan-januzaj-goal-england-v-belgium-match-45'

manual_calib = join(data_path, dataset, 'calib', '00114.npy')
calib_npy = np.load(manual_calib).item()

A, R, T = calib_npy['A'], calib_npy['R'], calib_npy['T']
h, w = 2160//2, 3840//2

img = cv2.imread(join(data_path, dataset, 'images', '00145.jpg'))[:, :, ::-1]
mask = cv2.imread(join(data_path, dataset, 'detectron', '00145.png'))

edges = utils.robust_edge_detection(img)
skel = medial_axis(edges, return_distance=False)
edges = skel.astype(np.uint8)

mask = cv2.dilate(mask[:, :, 0], np.ones((25, 25), dtype=np.uint8)) / 255

edges = edges * (1 - mask)
dist_transf = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 0)


template, field_mask = utils.draw_field(A, R, T, h, w)
II, JJ = (template > 0).nonzero()
synth_field2d = np.array([[JJ, II]]).T[:, :, 0]
field3d = utils.plane_points_to_3d(synth_field2d, A, R, T)

# xx, yy = np.meshgrid(np.arange(h), np.arange(w))
# f = interpolate.interp2d(np.arange(w), np.arange(h), dist_transf, kind='linear')


def _fun_distance_transform3(params_, dist_map_, points3d):
    theta_x_, theta_y_, theta_z_, fx_, tx_, ty_, tz_ = params_
    h_, w_ = dist_map_.shape[0:2]

    n_ = points3d.shape[0]

    cx_, cy_ = float(dist_map_.shape[1])/2.0, float(dist_map_.shape[0])/2.0

    R_ = utils.Rz(theta_z_).dot(utils.Ry(theta_y_)).dot(utils.Rx(theta_x_))
    A_ = np.eye(3, 3)
    A_[0, 0], A_[1, 1], A_[0, 2], A_[1, 2] = fx_, fx_, cx_, cy_

    T_ = np.array([[tx_], [ty_], [tz_]])

    p2_ = A_.dot(R_.dot(points3d.T) + np.tile(T_, (1, n_)))
    x, y, z = p2_[0, :], p2_[1, :], p2_[2, :]
    x = x/z
    y = y/z

    x = np.maximum(np.minimum(x, w_-1), 0).astype(int)
    y = np.maximum(np.minimum(y, h_-1), 0).astype(int)

    return np.sum(dist_map_[y, x])


theta_x, theta_y, theta_z = utils.get_angle_from_rotation(R)
fx, fy, cx, cy = A[0, 0], A[1, 1], A[0, 2], A[1, 2]

params = np.hstack((theta_x, theta_y, theta_z, fx, T[0, 0], T[1, 0], T[2, 0]))
start = time.time()
res_ = minimize(_fun_distance_transform3, params, args=(dist_transf, field3d), method='Nelder-Mead', options={'disp': True, 'maxiter': 10, 'maxfev':100})
end = time.time()
print('optim: {0:.4f}\n\n'.format(end-start))

result = res_.x

print(params)
print(result)

theta_x_, theta_y_, theta_z_, fx_, tx_, ty_, tz_ = result

cx_, cy_ = float(dist_transf.shape[1]) / 2.0, float(dist_transf.shape[0]) / 2.0

R__ = utils.Rz(theta_z_).dot(utils.Ry(theta_y_)).dot(utils.Rx(theta_x_))
T__ = np.array([[tx_], [ty_], [tz_]])
A__ = np.eye(3, 3)
A__[0, 0], A__[1, 1], A__[0, 2], A__[1, 2] = fx_, fx_, cx_, cy_


p0, _ = utils.project(vertices, A, R, T)
p1, _ = utils.project(vertices, A__, R__, T__)


p0, _ = utils.inside_frame(p0, h, w)
p1, _ = utils.inside_frame(p1, h, w)

plt.imshow(img)
plt.plot(p0[:, 0], p0[:, 1], '.')
plt.plot(p1[:, 0], p1[:, 1], '--')
plt.show()





# ===============================================
# Render with Pytorch

A_torch = torch.from_numpy(A)
R_torch = torch.from_numpy(R)
T_torch = torch.from_numpy(T)

vertices_torch = torch.from_numpy(vertices)

pcoords = A_torch@(R_torch@vertices_torch.transpose(0, 1)) + T_torch
X = pcoords[0, :]
Y = pcoords[1, :]
Z = pcoords[2, :].clamp(min=1e-3)


X_norm = X / Z  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
Y_norm = Y / Z  # Idem [B, H*W]

pixel_coords = torch.stack([X_norm, Y_norm], dim=1)  # [H*W, 2]

