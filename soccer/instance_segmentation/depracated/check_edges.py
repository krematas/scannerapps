import cv2
import numpy as np
from matplotlib import pyplot as plt

path_to_img = '/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/adnan-januzaj-goal-england-v-belgium-match-45/tmp/00001_0.jpg'
img = cv2.imread(path_to_img, 0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img/255.0,cv2.CV_64F,1,0,ksize=11)
sobely = cv2.Sobel(img/255.0,cv2.CV_64F,0,1,ksize=11)
mag = cv2.magnitude(sobelx, sobely)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(mag)


pd = cv2.ximgproc.createStructuredEdgeDetection('/home/krematas/code/scannerapps/edge_model.yml.gz')
img = cv2.imread(path_to_img)
edges = pd.detectEdges(img.astype(np.float32)/255.0)

ax[1].imshow(edges)
plt.show()