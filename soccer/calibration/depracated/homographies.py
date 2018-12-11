import soccer.calibration.utils as utils
import numpy as np
from os.path import join
import cv2
import matplotlib.pyplot as plt


data_path = '/home/krematas/Mountpoints/grail/data/Singleview/Soccer/'
path_to_field3d = join(data_path, 'field.ply')

vertex_data, _, _, _ = utils.read_ply(path_to_field3d)
vertices, _, _ = utils.ply_to_numpy(vertex_data)
n_vertices = vertices.shape[0]

dataset = 'Russia2018/antoine-griezmann-goal-uruguay-france'

manual_calib = join(data_path, dataset, 'calib', '00001.npy')
calib_npy = np.load(manual_calib).item()

A, R, T = calib_npy['A'], calib_npy['R'], calib_npy['T']
h, w = 2160//2, 3840//2

img = cv2.imread(join(data_path, dataset, 'images', '00001.jpg'))[:, :, ::-1]
mask = cv2.imread(join(data_path, dataset, 'detectron', '00001.png'))


def H_from_points(fp, tp):
    """ Find homography H, such that fp is mapped to tp
        using the linear DLT method. Points are conditioned
        automatically. """

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # condition points (important for numerical reasons)
    # --from points--
    m = np.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp = np.dot(C1, fp)

    # --to points--
    m = np.mean(tp[:2], axis=1)
    maxstd = max(np.std(tp[:2], axis=1)) + 1e-9
    C2 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp = np.dot(C2, tp)

    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = np.zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0,
                    tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1,
                        tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]

    U, S, V = np.linalg.svd(A)
    H = V[8].reshape((3, 3))

    # decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    # normalize and return
    return H / H[2, 2]




def _set_correspondences(img, field_img_path='/home/krematas/code/soccerontable/demo/data/field.png'):

    sfactor = 1.0
    field_img = cv2.imread(field_img_path)

    h2, w2 = field_img.shape[0:2]
    W, H = 104.73, 67.74

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.resize(img, None, fx=sfactor, fy=sfactor))
    ax[1].imshow(field_img)

    ax[0].axis('off')
    ax[1].axis('off')

    points2d = []
    points3d = []

    counter = 0

    def onclick(event):
        # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       (event.button, event.x, event.y, event.xdata, event.ydata))
        x, y = event.xdata, event.ydata
        if event.inaxes.axes.get_position().x0 < 0.5:
            ax[0].plot(x, y, 'r.', ms=10)
            points2d.append([x*sfactor, y*sfactor])
        else:
            ax[1].plot(x, y, 'b+', ms=10)
            points3d.append([x, y])
        plt.show()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    points2d = np.array(points2d)
    points3d = np.array(points3d)

    print(points2d)
    print(points3d)
    # points3d[:, 0] = ((points3d[:, 0] - w2 / 2.) / w2) * W
    # points3d[:, 2] = ((points3d[:, 2] - h2 / 2.) / h2) * H

    return points2d, points3d

# p_img, p_plane = _set_correspondences(img)


p_img = np.array([[676.96745103,  399.88873101],
                 [1625.03959532,  821.49685048],
                 [1255.85942853,  354.01427242],
                 [1771.40096322,  603.04704764]])
p_plane = np.array([[508.1655246,   78.61610007],
                    [508.1655246,  307.76311668],
                    [600.78482772,  78.61610007],
                    [571.96993342, 247.38905243]])

# p_img = np.array([[677.92316892, 435.62015256],
#                  [ 79.37070916, 219.35484776],
#                  [833.02252893, 292.53553171],
#                  [403.76866636, 186.58737734]])
# p_plane = np.array([[507.47945569, 310.50739233],
#                  [507.47945569, 79.9882379 ],
#                  [600.09875881, 248.07512134],
#                  [600.09875881, 78.61610007]])
#
W, H = 104.73, 67.74
h2, w2 = 390, 603
p_plane[:, 0] = ((p_plane[:, 0] - w2 / 2.) / w2) * W
p_plane[:, 1] = ((p_plane[:, 1] - h2 / 2.) / h2) * H

M, mask = cv2.findHomography(p_img, p_plane, cv2.RANSAC)
# M = H_from_points(p_img, p_plane)
# im_dst = cv2.warpPerspective(img, M, (1024, 1024))
# plt.imshow(im_dst)
# plt.show()

h1 = M[:, 0]
h2 = M[:, 1]

from sympy import *
from sympy.solvers.solveset import linsolve
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt

cx = w/2.
cy = h/2.
# f, cx, cy = symbols('f cx, cy')
# K = Matrix([[f, 0, cx], [0, f, cy], [0, 0, 1]])

f = symbols('f')
omega = Matrix([[1, 0, -cx], [0, 1, -cy], [-cx, -cy, f**2 + cx**2 + cy**2]])
h1_s = Matrix(h1)
h2_s = Matrix(h2)

f1 = h1_s.T*omega*h1_s-h2_s.T*omega*h2_s
f2 = h1_s.T*omega*h2_s

# solve(f1, f)
# solve(f2, f)
nsolve((f1, f2), f, 1000)
