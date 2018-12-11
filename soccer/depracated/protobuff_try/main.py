import soccer.depracated.protobuff_try.protofiles.image_pb2 as image_pb2
import cv2
import pickle

image = image_pb2.Image()


img = cv2.imread('/home/krematas/Downloads/untitled.png')

image.width = img.shape[1]
image.height = img.shape[0]
image.image_data = pickle.dumps(img)

data = image.SerializeToString()

with open('data.bin', 'wb') as f:
    f.write(data)
