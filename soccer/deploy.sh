export PROJECT=$(gcloud config get-value project)
echo $PROJECT

TAG=gpu-8.0-cudnn6-latest
docker pull scannerresearch/scanner:$TAG
docker build -f Dockerfile.master -t gcr.io/$PROJECT/scanner-master:$TAG .
docker build -f Dockerfile.worker -t gcr.io/$PROJECT/scanner-worker:$TAG .

gcloud docker -- push gcr.io/$PROJECT/scanner-master:$TAG
gcloud docker -- push gcr.io/$PROJECT/scanner-worker:$TAG

kubectl delete deploy --all
kubectl create -f master.yml
kubectl create -f worker.yml

kubectl expose deploy/scanner-master --type=NodePort --port=8080
