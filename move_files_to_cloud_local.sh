#!/usr/bin/env bash
dataset=$1
cloud_instance=$2

PATH_TO_DATA=/home/krematas/Mountpoints/grail/data/Singleview/Soccer

gcloud compute scp $PATH_TO_DATA/$dataset/images/*.jpg krematas@$cloud_instance:/home/krematas/data/$dataset/images --zone us-west1-b
gcloud compute scp $PATH_TO_DATA/$dataset/detectron/*.png krematas@$cloud_instance:/home/krematas/data/$dataset/detectron --zone us-west1-b
gcloud compute scp $PATH_TO_DATA/$dataset/calib/00001.npy krematas@$cloud_instance:/home/krematas/data/$dataset/calib/ --zone us-west1-b
gcloud compute scp $PATH_TO_DATA/$dataset/metadata/*.p krematas@$cloud_instance:/home/krematas/data/$dataset/metadata --zone us-west1-b