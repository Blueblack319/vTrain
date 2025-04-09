docker run \
    --gpus all \
    -it \
    --name vtrain_nemo \
    --rm \
    -v /home2/jaehoon/vTrain:/workspace/vTrain \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit \
    stack=67108864 \
    nvcr.io/nvidia/nemo:25.02
