import scannerpy
import cv2
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType

import numpy as np
import pickle


@scannerpy.register_python_op()
class CropPlayersClass(scannerpy.Kernel):

    def __init__(self, config):
        self.w = config.args['w']
        self.h = config.args['h']
        self.limps = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 11], [11, 12], [12, 13], [1, 8],
             [8, 9], [9, 10], [14, 15], [16, 17], [0, 14], [0, 15], [14, 16], [15, 17]])

    def execute(self, image: FrameType, mask: FrameType, metadata: bytes) -> FrameType:

        # output = np.zeros((self.h, self.w, 3), dtype=np.float32)
        output = image.copy()
        metadata = pickle.loads(metadata)
        poses = metadata['poses']
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

        # mask = mask[:, :, 0]/255.0
        # output[:, :, 0] *= mask
        # output[:, :, 1] *= mask
        # output[:, :, 2] *= mask

        output = output * (mask / 255.)

        return output.astype(np.uint8)