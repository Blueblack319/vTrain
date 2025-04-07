docker run \
  --gpus all \
  -it \
  --rm \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v .:/workspace/vTrain \
  nvcr.io/nvidia/pytorch:24.10-py3
