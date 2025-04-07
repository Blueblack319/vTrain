#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

VOCAB_FILE="gpt2-vocab.json" #<Specify path to file>/gpt2-vocab.json
MERGE_FILE="gpt2-merges.txt" #<Specify path to file>/gpt2-merges.txt

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 1024
    --num-attention-heads 16 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 16 
    --global-batch-size 16 
    --train-iters 50
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    #--data-path $DATA_PATH 
    --mock-data
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 10,0,0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    # --save-interval 10000 
    # --eval-interval 500 
    # --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 16
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --profile
    --profile-step-start 8
    --profile-step-end 9
    --profile-ranks 0
)

CUDA_VISIBLE_DEVICES=3 python test_model_v2.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
