#!/bin/bash

echo "=== SASRec CE Full Vocab Pretraining (CDs) ==="

python train.py \
    --dataset amazon_cds \
    --run_name v3 \
    --maxlen 50 \
    --hidden_units 50 \
    --num_heads 1 \
    --num_blocks 2 \
    --dropout_rate 0.5 \
    --batch_size 128 \
    --lr 0.001 \
    --num_epochs 1000 \
    --patience 10 \
    --eval_interval 10 \
    --seed 42 \
    --device cuda

echo "=== Done ==="