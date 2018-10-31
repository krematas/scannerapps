import scannerpy
import cv2
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType

import numpy as np
import pickle


def project(points3d, A, R, T, scale_factor=1.0, dtype=np.int32):
    """ Project a set of 3D points (Nx3 or 3XN) to a camera with parameters A, R T.
    Return the pixel coordinates and its corresponding depth
    """

    if points3d.shape[0] != 3:
        points3d = points3d.T

    assert(T.shape == (3, 1))

    n_points = points3d.shape[1]

    projected_points_ = A.dot(R.dot(points3d) + np.tile(T, (1, n_points)))
    depth = projected_points_[2, :]
    pixels = projected_points_[0:2, :] / projected_points_[2, :] / scale_factor

    if issubclass(dtype, np.integer):
        pixels = np.round(pixels)

    pixels = np.array(pixels.T, dtype=dtype)

    return pixels, depth


def inside_frame(points2d, height, width, margin=0):
    valid = np.logical_and(np.logical_and(points2d[:, 0] >= 0+margin, points2d[:, 0] < width-margin),
                           np.logical_and(points2d[:, 1] >= 0+margin, points2d[:, 1] < height-margin))
    points2d = points2d[valid, :]
    return points2d, valid


def sample_spherical(npoints, ndim=3):
    np.random.seed(42)

    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def get_box_from_3d_shpere(A, R, T, h, w, center3d):
    sphere = sample_spherical(100) * 1.2

    bounding_shpere = sphere.T + center3d
    sphere2d, _ = project(bounding_shpere, A, R, T, dtype=int)
    sphere2d, _ = inside_frame(sphere2d, h, w)

    tmp_mask = np.zeros((h, w), dtype=np.float32)
    tmp_mask[sphere2d[:, 1], sphere2d[:, 0]] = 1

    # center2d, _ = cam.project(np.array([center3d]), dtype=int)
    # img[center2d[0, 1], center2d[:, 0], 0] = 1
    # io.imshow(img)

    contours, hierarchy, _ = cv2.findContours(tmp_mask.astype(np.uint8), 1, 2)
    x_, y_, ww_, hh_ = cv2.boundingRect(contours)
    box = np.array([x_, y_, x_ + ww_, y_ + hh_, 1])

    return box


def unproject(A, R, T, points2d, depth):
    if points2d.shape[0] != 2:
        points2d = points2d.T

    n_points = points2d.shape[1]

    points2d = np.vstack((points2d, np.ones(points2d.shape[1])))
    pixel_i = np.linalg.inv(A).dot(points2d)
    pixel_world = R.T.dot(np.multiply(depth, pixel_i) - np.tile(T, (1, n_points)))

    return pixel_world


def get_camera_position(R, T):
    return -R.T.dot(T)


def ray_plane_intersection(ray_origin, ray_dir, plane_origin, plane_normal):
    n_rays = ray_dir.shape[0]
    denom = np.inner(plane_normal, ray_dir)
    p0l0 = plane_origin - ray_origin
    t = np.divide(np.inner(p0l0, plane_normal), denom)
    point3d = np.tile(ray_origin, (n_rays, 1)) + np.multiply(np.tile(t, (3, 1)).T, ray_dir)
    return point3d


def lift_keypoints_in_3d(A, R, T, keypoints, pad=0):
    """
    cam: camera class
    points: Nx3 matrix, N number of keypoints and X, Y, score
    Assumes that the lowest of the point touces the ground
    """

    # Make a bounding box
    x1, y1, x2, y2 = min(keypoints[:, 0])-pad, min(keypoints[:, 1])-pad, max(keypoints[:, 0])+pad, max(keypoints[:, 1])+pad
    bbox = np.array([[x1, y2], [x2, y2], [x1, y1], [x2, y1]])

    bbox_camplane = unproject(A, R, T, bbox, 0.5)
    origin = get_camera_position(R, T).T
    bbox_direction = bbox_camplane.T - np.tile(origin, (bbox_camplane.shape[1], 1))
    bbox_direction /= np.tile(np.linalg.norm(bbox_direction, axis=1)[:, np.newaxis], (1, 3))
    bbox_onground = ray_plane_intersection(origin, bbox_direction, np.array([0, 0, 0]), np.array([0, 1, 0]))

    # Find the billboard plane
    p0 = bbox_onground[0, :]
    p1 = bbox_onground[1, :]
    p3_ = bbox_onground[0, :].copy()
    p3_[1] = 1.0  # Just a bit lifted, since we do not know the original extend

    billboard_n = np.cross(p1 - p0, p3_ - p0)

    # Find the pixels that are masked and contained in the bbox
    keypoints_camplane = unproject(A, R, T, keypoints[:, :2], 0.5)
    kp_direction = keypoints_camplane.T - np.tile(origin, (keypoints_camplane.shape[1], 1))
    kp_direction /= np.tile(np.linalg.norm(kp_direction, axis=1)[:, np.newaxis], (1, 3))
    kepoints_lifted = ray_plane_intersection(origin, kp_direction, p0, billboard_n)

    return kepoints_lifted


def get_poseimg_for_opt(sel_pose, poseimg, init_mask, n_bg=50):

    h, w = init_mask.shape[:2]
    bg_label = 1
    output = np.zeros((h, w, 3), dtype=np.float32) - 1
    II, JJ = (poseimg > 0).nonzero()
    Isel, J_sel = (poseimg == sel_pose).nonzero()

    output[II, JJ] = 0
    output[Isel, J_sel] = 2

    init_mask[Isel, J_sel] = 1
    # Sample also from points in the field
    init_mask = cv2.dilate(init_mask, np.ones((25, 25), np.uint8), iterations=1)

    I_bg, J_bg = (init_mask == 0).nonzero()
    rand_index = np.random.permutation(len(I_bg))[:n_bg]
    bg_points = np.array([J_bg[rand_index], I_bg[rand_index]]).T

    for k in range(bg_points.shape[0]):
        cv2.circle(output, (int(bg_points[k, 0]), int(bg_points[k, 1])), 2, (bg_label, 0, 0), -1)

    return output[:, :, 0]


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep


def refine_poses(poses, A, R, T, keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4, margin=0.0):
    W, H = 104.73, 67.74
    # remove the poses with few keypoints or they
    keep = []
    for ii in range(len(poses)):
        keypoints = poses[ii]
        valid = (keypoints[:, 2] > 0.).nonzero()[0]
        score = np.sum(keypoints[valid, 2])

        if len(valid) > keypoint_thresh and score > score_thresh and keypoints[1, 2] > neck_thresh:
            keep.append(ii)

    poses = [poses[ii] for ii in keep]

    root_part = 1
    root_box = []
    for ii in range(len(poses)):
        root_tmp = poses[ii][root_part, :]
        valid_keypoints = (poses[ii][:, 2] > 0).nonzero()
        root_box.append(
            [root_tmp[0] - 10, root_tmp[1] - 10, root_tmp[0] + 10, root_tmp[1] + 10,
             np.sum(poses[ii][valid_keypoints, 2])])
    root_box = np.array(root_box)

    # Perform Neck NMS
    if len(root_box.shape) == 1:
        root_box = root_box[None, :]
        keep2 = [0]
    else:
        keep2 = nms(root_box.astype(np.float32), 0.1)

    poses = [poses[ii] for ii in keep2]

    # Remove poses outside of field
    keep3 = []
    for ii in range(len(poses)):
        kp3 = lift_keypoints_in_3d(A, R, T, poses[ii])
        if (-W / 2. - margin) <= kp3[1, 0] <= (W / 2. + margin) and (-H / 2. - margin) <= kp3[1, 2] <= (
                H / 2. + margin):
            keep3.append(ii)

    poses = [poses[ii] for ii in keep3]
    return poses


@scannerpy.register_python_op()
class CropPlayersClass(scannerpy.Kernel):

    def __init__(self, config):
        self.w = config.args['w']
        self.h = config.args['h']
        self.margin = config.args['margin']
        self.limps = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 11], [11, 12], [12, 13], [1, 8],
             [8, 9], [9, 10], [14, 15], [16, 17], [0, 14], [0, 15], [14, 16], [15, 17]])

    def execute(self, image: FrameType, mask: FrameType, metadata: bytes) -> bytes:

        metadata = pickle.loads(metadata)
        poses = metadata['poses']
        A, R, T = metadata['A'], metadata['R'], metadata['T']

        poses = refine_poses(poses, A, R, T)
        # ==============================================================================================================
        # Make pose image
        pose_img = np.zeros((self.h, self.w, 3), dtype=np.float32)-1
        for i in range(len(poses)):
            keypoints = poses[i]

            lbl = i
            for k in range(self.limps.shape[0]):
                kp1, kp2 = self.limps[k, :].astype(int)
                bone_start = keypoints[kp1, :]
                bone_end = keypoints[kp2, :]
                bone_start[0] = np.maximum(np.minimum(bone_start[0], self.w - 1), 0.)
                bone_start[1] = np.maximum(np.minimum(bone_start[1], self.h - 1), 0.)

                bone_end[0] = np.maximum(np.minimum(bone_end[0], self.w - 1), 0.)
                bone_end[1] = np.maximum(np.minimum(bone_end[1], self.h - 1), 0.)

                if bone_start[2] > 0.0:
                    pose_img[int(bone_start[1]), int(bone_start[0])] = 1
                    cv2.circle(pose_img, (int(bone_start[0]), int(bone_start[1])), 2, (lbl, 0, 0), -1)

                if bone_end[2] > 0.0:
                    pose_img[int(bone_end[1]), int(bone_end[0])] = 1
                    cv2.circle(pose_img, (int(bone_end[0]), int(bone_end[1])), 2, (lbl, 0, 0), -1)

                if bone_start[2] > 0.0 and bone_end[2] > 0.0:
                    cv2.line(pose_img, (int(bone_start[0]), int(bone_start[1])), (int(bone_end[0]), int(bone_end[1])),
                             (lbl, 0, 0), 1)

        # ==============================================================================================================
        # estimate extent of players
        mask = mask[:, :, 0]/255
        pose_img = pose_img[:, :, 0]

        out = []
        for i in range(len(poses)):
            valid = poses[i][:, 2] > 0

            kp3 = lift_keypoints_in_3d(A, R, T, poses[i][valid, :], pad=0)

            center3d = np.mean(kp3, axis=0)
            # Most of keypoitns are in the upper body so the center of the mass is closer to neck
            center3d[1] -= 0.25

            _, center_depth = project(np.array([center3d]), A, R, T)

            bbox = get_box_from_3d_shpere(A, R, T, self.h, self.w, center3d)
            x1, y1, x2, y2 = bbox[:4]

            x1 -= self.margin
            y1 -= self.margin
            x2 += self.margin
            y2 += self.margin
            x1, x2, y1, y2 = max(x1, 0), min(self.w, x2), max(y1, 0), min(self.h, y2)

            img_crop = image[y1:y2, x1:x2, :]
            pose_img_crop = get_poseimg_for_opt(i, pose_img[y1:y2, x1:x2], mask[y1:y2, x1:x2], n_bg=30)+1
            mask_crop = mask[y1:y2, x1:x2]

            out.append({'img': img_crop, 'pose_img': pose_img_crop, 'mask': mask_crop})

        return pickle.dumps(out)


@scannerpy.register_python_op(name='BinaryToFrame')
def detectron_vizualize(config,
                        data: bytes) -> FrameType:
        data = pickle.loads(data)
        return data['img']