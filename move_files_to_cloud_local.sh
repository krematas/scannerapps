#!/usr/bin/env bash
dataset=$1
cloud_instance=$2

PATH_TO_DATA=/home/krematas/Mountpoints/grail/data/Singleview/Soccer

gcloud compute scp $PATH_TO_DATA/$dataset/images/*.jpg krematas@$cloud_instance:/home/krematas/data/$dataset/images --zone us-west1-b
gcloud compute scp $PATH_TO_DATA/$dataset/detectron/*.jpg krematas@$cloud_instance:/home/krematas/data/$dataset/images --zone us-west1-b
gcloud compute scp $PATH_TO_DATA/$dataset/images/*.jpg krematas@$cloud_instance:/home/krematas/data/$dataset/images --zone us-west1-b