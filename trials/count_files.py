import glob
import os


path_to_data = '/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018'

goal_dirs = [item for item in os.listdir(path_to_data) if os.path.isdir(os.path.join(path_to_data, item))]
goal_dirs.sort()

for i in range(len(goal_dirs)):
    count = len(glob.glob(os.path.join(path_to_data, goal_dirs[i], 'images', '*.jpg')))
    print(i, goal_dirs[i], count)