import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/krematas/code/openpose/python/openpose/python')
from openpose import *

params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = '/home/krematas/code/openpose/models'
# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)


img = cv2.imread("/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/antoine-griezmann-goal-uruguay-france/images/00001.jpg")
keypoints, output_image = openpose.forward(img, True)
# Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
print(keypoints)
# Display the image
plt.imshow("output", output_image)
plt.show()
