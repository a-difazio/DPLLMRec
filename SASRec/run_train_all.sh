#!/bin/bash

echo "=== Pretraining SASRec CE Full Vocab ==="

# ML-1M
python train.py \
    --dataset movielens \
    --run_name user_stratified_split \
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
    --seed 42

# Music
python train.py \
    --dataset amazon_music \
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