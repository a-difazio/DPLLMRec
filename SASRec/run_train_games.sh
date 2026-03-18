#!/bin/bash

echo "=== Amazon Games ==="

# Games
python train.py \
    --dataset amazon_games \
    --run_name user_stratified_split \
    --maxlen 50 \
    --hidden_units 128 \
    --num_heads 2 \
    --num_blocks 2 \
    --dropout_rate 0.2 \
    --batch_size 256 \
    --lr 0.001 \
    --num_epochs 1000 \
    --patience 10 \
    --eval_interval 10 \
    --seed 42

echo "=== Done ==="