#!/bin/bash

echo "=== SASRec CE Full Vocab Pretraining (CDs) ==="

python train.py \
    --dataset amazon_cds \
    --run_name maxlen50_final \
    --maxlen 50 \
    --hidden_units 50 \
    --num_heads 1 \
    --num_blocks 1 \
    --dropout_rate 0.5 \
    --batch_size 256 \
    --lr 0.0001 \
    --l2_emb 1e-4 \
    --num_epochs 1000 \
    --patience 10 \
    --eval_interval 10 \
    --seed 42 \
    --device cuda

echo "=== Done ==="
