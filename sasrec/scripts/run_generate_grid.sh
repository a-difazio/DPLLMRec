#!/bin/bash

DATASET=$1
GPU=$2

if [ -z "$DATASET" ] || [ -z "$GPU" ]; then
  echo "Usage: $0 <dataset> <gpu>"
  exit 1
fi

echo "Running GRID on dataset: $DATASET | GPU: $GPU"

TOP_PS=(0.2 0.4 0.6 0.8 1)
TEMPERATURES=(1.0 0.9 0.85 1.15 0.7)
PENALTIES=(1.05 1.1 1.2 1.3)

for TEMP in "${TEMPERATURES[@]}"; do
  for TOP_P in "${TOP_PS[@]}"; do
    for PEN in "${PENALTIES[@]}"; do

      GEN_NAME="temp${TEMP}_topp${TOP_P}_pen${PEN}"

      echo "Running: $GEN_NAME"

      python generate.py \
        --dataset "$DATASET" \
        --run_name final \
        --gen_name "$GEN_NAME" \
        --checkpoint checkpoint_best.pth \
        --device cuda:$GPU \
        --temperature $TEMP \
        --top_p $TOP_P \
        --penalty $PEN \
        --seed 42 \
        > "${DATASET}_${GEN_NAME}.log" 2>&1

      echo "Done: $GEN_NAME"

    done
  done
done

echo "=== GRID COMPLETED for $DATASET ==="