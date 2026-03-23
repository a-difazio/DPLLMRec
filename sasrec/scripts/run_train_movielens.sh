#!/bin/bash

echo "=== SASRec CE Full Vocab Pretraining (MovieLens) ==="

python train.py \
    --dataset movielens \
    --run_name final \
    --maxlen 200 \
    --hidden_units 50 \
    --num_heads 1 \
    --num_blocks 2 \
    --dropout_rate 0.2 \
    --batch_size 128 \
    --lr 0.001 \
    --num_epochs 1000 \
    --patience 5 \
    --eval_interval 10 \
    --seed 42 \
    --device cuda

echo "=== Done ==="
