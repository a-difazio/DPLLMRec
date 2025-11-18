#!/bin/bash

set -e

DATASET="ml-1m"
TRAIN_DIR="default"
DEVICE="cuda"
MAXLEN=200
HIDDEN_UNITS=50
NUM_BLOCKS=2
NUM_HEADS=1
DROPOUT_RATE=0.2
L2_EMB=0.0
EVAL_INTERVAL=20

echo "Starting Experiment 1"
python train.py \
    --dataset $DATASET \
    --train_dir ${TRAIN_DIR} \
    --lr 0.001 \
    --batch_size 128 \
    --maxlen $MAXLEN \
    --hidden_units $HIDDEN_UNITS \
    --num_blocks $NUM_BLOCKS \
    --num_heads $NUM_HEADS \
    --dropout_rate $DROPOUT_RATE \
    --l2_emb $L2_EMB \
    --device $DEVICE \
    --num_epochs 100 \
    --eval_interval $EVAL_INTERVAL

echo "Experiment 1 Finished."