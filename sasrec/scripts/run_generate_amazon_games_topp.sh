#!/bin/bash

DEVICE=${1:-cuda}

echo "=== Generating for amazon_games (device: $DEVICE) ==="

TOP_PS=(0.1 0.2 0.3 0.5 0.6 0.7 0.8 0.9 0.95 1)

for TOP_P in "${TOP_PS[@]}"
do
    GEN_NAME="temp1.0_topp${TOP_P}_topknone_pen0.0"

    echo "Running $GEN_NAME"

    python generate.py \
        --dataset amazon_games \
        --run_name final \
        --gen_name $GEN_NAME \
        --checkpoint checkpoint_best.pth \
        --device $DEVICE \
        --temperature 1.0 \
        --penalty 0.0 \
        --seed 42 \
        --top_p $TOP_P

    echo "=== Done $GEN_NAME ==="
done

echo "=== All experiments completed ==="