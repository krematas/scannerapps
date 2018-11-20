import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from scipy.optimize import minimize


W, H = 104.73, 67.74


def make_field_circle(r=9.15, nn=1):
    """
    Returns points that lie on a circle on the ground
    :param r: radius
    :param nn: points per arc?
    :return: 3D points on a circle with y = 0
    """
    d = 2 * np.pi * r
    n = int(nn * d)
    return [(np.cos(2 * np.pi / n * x) * r, 0, np.sin(2 * np.pi / n * x) * r) for x in range(0, n + 1)]


def ray_plane_intersection(ray_origin, ray_dir, plane_origin, plane_normal):
    n_rays = ray_dir.shape[0]
    denom = np.inner(plane_normal, ray_dir)
    p0l0 = plane_origin - ray_origin
    t = np.divide(np.inner(p0l0, plane_normal), denom)
    point3d = np.tile(ray_origin, (n_rays, 1)) + np.multiply(np.tile(t, (3, 1)).T, ray_dir)
    return point3d


def get_field_points():

    outer_rectangle = np.array([[-W / 2., 0, H / 2.],
                                [-W / 2., 0, -H / 2.],
                                [W / 2., 0, -H / 2.],
                                [W / 2., 0, H / 2.]])

    mid_line = np.array([[0., 0., H / 2],
                        [0., 0., -H / 2]])

    left_big_box = np.array([[-W / 2., 0, 40.32/2.],
                             [-W / 2., 0, -40.32 / 2.],
                             [-W / 2. + 16.5, 0, -40.32 / 2.],
                             [-W/2.+16.5, 0, 40.32/2.]])

    right_big_box = np.array([[W/2.-16.5, 0, 40.32/2.],
                             [W/2., 0, 40.32/2.],
                             [W/2., 0, -40.32/2.],
                             [W/2.-16.5, 0, -40.32/2.]])

    left_small_box = np.array([[-W/2., 0, 18.32/2.],
                               [-W / 2., 0, -18.32 / 2.],
                               [-W / 2. + 5.5, 0, -18.32 / 2.],
                               [-W/2.+5.5, 0, 18.32/2.] ])

    right_small_box = np.array([[W/2.-5.5, 0, 18.32/2.],
                               [W/2., 0, 18.32/2.],
                               [W/2., 0, -18.32/2.],
                               [W/2.-5.5, 0, -18.32/2.]])

    central_circle = np.array(make_field_circle(r=9.15, nn=1))

    left_half_circile = np.array(make_field_circle(9.15))
    left_half_circile[:, 0] = left_half_circile[:, 0] - W / 2. + 11.0
    index = left_half_circile[:, 0] > (-W / 2. + 16.5)
    left_half_circile = left_half_circile[index, :]

    right_half_circile = np.array(make_field_circle(9.15))
    right_half_circile[:, 0] = right_half_circile[:, 0] + W / 2. - 11.0
    index = right_half_circile[:, 0] < (W / 2. - 16.5)
    right_half_circile = right_half_circile[index, :]

    return [outer_rectangle, left_big_box, right_big_box, left_small_box, right_small_box,
            left_half_circile, right_half_circile, central_circle, mid_line]


def robust_edge_detection(img):
    # Find edges
    kernel_size = 5
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = cv2.Canny((blur_gray * 255).astype(np.uint8), 10, 200, apertureSize=5)

    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(edges)[0]  # Position 0 of the returned tuple are the detected lines

    long_lines = []
    for j in range(lines.shape[0]):
        x1, y1, x2, y2 = lines[j, 0, :]
        if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) > 50:
            long_lines.append(lines[j, :, :])

    lines = np.array(long_lines)
    edges = 1 * np.ones_like(img)
    drawn_img = lsd.drawSegments(edges, lines)
    edges = (drawn_img[:, :, 2] > 1).astype(np.float32)

    kernel = np.ones((7, 7), np.uint8)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    return edges


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


def unproject(points2d, depth, A, R, T, scale_factor=1):

    A_i = np.linalg.inv(A)
    if points2d.shape[0] != 2:
        points2d = points2d.T

    n_points = points2d.shape[1]

    points2d = np.vstack((points2d*scale_factor, np.ones(points2d.shape[1])))
    pixel_i = A_i.dot(points2d)
    pixel_world = R.T.dot(np.multiply(depth, pixel_i) - np.tile(T, (1, n_points)))

    return pixel_world


def plane_points_to_3d(points2d, A, R, T, plane_origin=np.array([0, 0, 0]), plane_direction=np.array([0, 1, 0])):
    p3 = unproject(points2d, 0.5, A, R, T)
    origin = -R.T.dot(T).T
    direction = p3.T - np.tile(origin, (p3.shape[1], 1))
    direction /= np.tile(np.linalg.norm(direction, axis=1)[:, np.newaxis], (1, 3))
    plane3d = ray_plane_intersection(origin, direction, plane_origin, plane_direction)
    return plane3d


def project_field_to_image(A, R, T):

    field_list = get_field_points()

    field_points2d = []
    for i in range(len(field_list)):
        tmp, depth = project(field_list[i], A, R, T)

        behind_points = (depth < 0).nonzero()[0]
        tmp[behind_points, :] *= -1
        field_points2d.append(tmp)

    return field_points2d


def draw_field(A, R, T, h, w):

    field_points2d = project_field_to_image(A, R, T)

    # Check if the entities are 7
    assert len(field_points2d) == 9

    img_polygon = Polygon([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw the boxes
    for i in range(5):

        # And make a new image with the projected field
        linea = LineString([(field_points2d[i][0, :]),
                            (field_points2d[i][1, :])])

        lineb = LineString([(field_points2d[i][1, :]),
                            (field_points2d[i][2, :])])

        linec = LineString([(field_points2d[i][2, :]),
                            (field_points2d[i][3, :])])

        lined = LineString([(field_points2d[i][3, :]),
                            (field_points2d[i][0, :])])

        if i == 0:
            polygon0 = Polygon([(field_points2d[i][0, :]),
                                (field_points2d[i][1, :]),
                                (field_points2d[i][2, :]),
                                (field_points2d[i][3, :])])

            intersect0 = img_polygon.intersection(polygon0)
            if not intersect0.is_empty:
                pts = np.array(list(intersect0.exterior.coords), dtype=np.int32)
                pts = pts[:, :].reshape((-1, 1, 2))
                cv2.fillConvexPoly(mask, pts, (255, 255, 255))

        intersect0 = img_polygon.intersection(linea)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            if pts.shape[0] < 2:
                continue
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255))

        intersect0 = img_polygon.intersection(lineb)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            if pts.shape[0] < 2:
                continue
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255))

        intersect0 = img_polygon.intersection(linec)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            if pts.shape[0] == 2:
                cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255))

        intersect0 = img_polygon.intersection(lined)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            if pts.shape[0] < 2:
                continue
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255))

    # Mid line
    line1 = LineString([(field_points2d[8][0, :]),
                        (field_points2d[8][1, :])])

    intersect1 = img_polygon.intersection(line1)
    if not intersect1.is_empty:
        pts = np.array(list(list(intersect1.coords)), dtype=np.int32)
        pts = pts[:, :].reshape((-1, 1, 2))
        cv2.fillConvexPoly(canvas, pts, (255, 255, 255), )

    # Circles
    for ii in range(5, 8):
        for i in range(field_points2d[ii].shape[0] - 1):
            line2 = LineString([(field_points2d[ii][i, :]),
                                (field_points2d[ii][i + 1, :])])
            intersect2 = img_polygon.intersection(line2)
            if not intersect2.is_empty:
                pts = np.array(list(list(intersect2.coords)), dtype=np.int32)
                pts = pts[:, :].reshape((-1, 1, 2))
                cv2.fillConvexPoly(canvas, pts, (255, 255, 255), )

    return canvas[:, :, 0] / 255., mask[:, :, 0] / 255.


def Rx(theta):
    theta = np.deg2rad(theta)
    rcos = np.cos(theta)
    rsin = np.sin(theta)
    A = np.array([[1, 0, 0],
                  [0, rcos, -rsin],
                  [0, rsin, rcos]])
    return A


def Ry(theta):
    theta = np.deg2rad(theta)
    rcos = np.cos(theta)
    rsin = np.sin(theta)
    A = np.array([[rcos, 0, rsin],
                  [0, 1, 0],
                  [-rsin, 0, rcos]])
    return A


def Rz(theta):
    theta = np.deg2rad(theta)
    rcos = np.cos(theta)
    rsin = np.sin(theta)
    A = np.array([[rcos, -rsin, 0],
                  [rsin, rcos, 0],
                  [0, 0, 1]])
    return A


def get_angle_from_rotation(R):

    M = np.asarray(R)
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat

    if abs(r31) != 1:
        y = -np.arcsin(r31)
        y2 = np.pi-y
        x2 = np.arctan2(r32 / np.cos(y2), r33 / np.cos(y2))
        z2 = np.arctan2(r21 / np.cos(y2), r11 / np.cos(y2))

        y = y2
        x = x2
        z = z2
    else:
        z = 0
        if r31 == -1:
            y = np.pi/2.
            x = np.arctan2(r12, r13)
        else:
            y = -np.pi / 2.
            x = np.arctan2(-r12, -r13)

    return np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)


def inside_frame(points2d, height, width, margin=0):
    valid = np.logical_and(np.logical_and(points2d[:, 0] >= 0+margin, points2d[:, 0] < width-margin),
                           np.logical_and(points2d[:, 1] >= 0+margin, points2d[:, 1] < height-margin))
    points2d = points2d[valid, :]
    return points2d, valid


def _fun_distance_transform(params_, dist_map_, points3d):
    theta_x_, theta_y_, theta_z_, fx_, tx_, ty_, tz_ = params_
    h_, w_ = dist_map_.shape[0:2]
    n_ = points3d.shape[0]

    cx_, cy_ = float(dist_map_.shape[1])/2.0, float(dist_map_.shape[0])/2.0

    R_ = Rz(theta_z_).dot(Ry(theta_y_)).dot(Rx(theta_x_))
    A_ = np.eye(3, 3)
    A_[0, 0], A_[1, 1], A_[0, 2], A_[1, 2] = fx_, fx_, cx_, cy_

    T_ = np.array([[tx_], [ty_], [tz_]])

    p2_ = A_.dot(R_.dot(points3d.T) + np.tile(T_, (1, n_)))
    p2_ /= p2_[2, :]
    p2_ = p2_.T[:, 0:2]
    p2_ = np.round(p2_).astype(int)
    _, valid_id_ = inside_frame(p2_, h_, w_)

    residual = np.zeros((n_,)) + 0.0
    residual[valid_id_] = dist_map_[p2_[valid_id_, 1], p2_[valid_id_, 0]]
    return np.sum(residual)


def calibrate_camera_dist_transf(A, R, T, dist_transf, points3d):

    theta_x, theta_y, theta_z = get_angle_from_rotation(R)
    fx, fy, cx, cy = A[0, 0], A[1, 1], A[0, 2], A[1, 2]

    params = np.hstack((theta_x, theta_y, theta_z, fx, T[0, 0], T[1, 0], T[2, 0]))

    res_ = minimize(_fun_distance_transform, params, args=(dist_transf, points3d),
                    method='Powell', options={'disp': False, 'maxiter': 5000})
    result = res_.x

    theta_x_, theta_y_, theta_z_, fx_, tx_, ty_, tz_ = result

    cx_, cy_ = float(dist_transf.shape[1]) / 2.0, float(dist_transf.shape[0]) / 2.0

    R__ = Rz(theta_z_).dot(Ry(theta_y_)).dot(Rx(theta_x_))
    T__ = np.array([[tx_], [ty_], [tz_]])
    A__ = np.eye(3, 3)
    A__[0, 0], A__[1, 1], A__[0, 2], A__[1, 2] = fx_, fx_, cx_, cy_

    return A__, R__, T__
