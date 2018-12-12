#!/usr/bin/env bash
export ZONE=us-west1-b

CLUSTER_NAME=soccer-cluster
gcloud beta container clusters create $CLUSTER_NAME \
       --zone "$ZONE" \
       --machine-type "n1-standard-32" \
       --num-nodes 1 \
       --cluster-version 1.9

gcloud container clusters get-credentials $CLUSTER_NAME --zone "$ZONE"

gcloud beta container node-pools create workers \
       --zone "$ZONE" \
       --cluster $CLUSTER_NAME \
       --machine-type "n1-standard-32" \
       --num-nodes 1 \
       --enable-autoscaling \
       --min-nodes 0 \
       --max-nodes 10 \
       --preemptible

kubectl create secret generic aws-storage-key \
    --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

export PROJECT=$(gcloud config get-value project)
echo $PROJECT

sed "s/YOUR_PROJECT_ID/$PROJECT/g" master.yml.template > master.yml
sed "s/YOUR_PROJECT_ID/$PROJECT/g" worker.yml.template > worker.yml

#kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml
