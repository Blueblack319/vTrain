CUDA_VISIBLE_DEVICES=3 nsys profile -s none \
    -o test_nemo_gpt \
    -t cuda,nvtx \
    --gpu-metrics-devices=cuda-visible \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite true \
    python ./test_nemo_gpt.py
    # --cudabacktrace=all \