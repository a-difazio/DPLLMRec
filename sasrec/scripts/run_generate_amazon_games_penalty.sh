#!/bin/bash

DEVICE=${1:-cuda}

echo "=== Generating for amazon_games (device: $DEVICE) ==="

PENALTIES=(1.0 1.05 1.1 1.2 1.3)

for PENALTY in "${PENALTIES[@]}"
do
    GEN_NAME="temp1.0_topp0.5_topknone_pen${PENALTY}"

    echo "Running $GEN_NAME"

    python generate.py \
        --dataset amazon_games \
        --run_name final \
        --gen_name $GEN_NAME \
        --checkpoint checkpoint_best.pth \
        --device $DEVICE \
        --temperature 1.0 \
        --penalty $PENALTY \
        --seed 42 \
        --top_p 0.5

    echo "=== Done $GEN_NAME ==="
done

echo "=== All experiments completed ==="