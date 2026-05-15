#!/bin/bash

DEVICE=${1:-cuda}

echo "=== Generating for amazon_games (device: $DEVICE) ==="

TEMPS=(0.7 0.85 0.9 1.0 1.15)

for TEMP in "${TEMPS[@]}"
do
    GEN_NAME="temp${TEMP}_topp1.0_topknone_pen1.0"

    echo "Running $GEN_NAME"

    python generate.py \
        --dataset amazon_games \
        --run_name final \
        --gen_name $GEN_NAME \
        --checkpoint checkpoint_best.pth \
        --device $DEVICE \
        --temperature $TEMP \
        --penalty 1.0 \
        --seed 42 \
        --top_p 1.0

    echo "=== Done $GEN_NAME ==="
done

echo "=== All experiments completed ==="