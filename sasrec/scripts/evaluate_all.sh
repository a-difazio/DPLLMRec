#!/bin/bash

echo "=== Running evaluations ==="

run_eval () {
  dataset=$1
  run_name=$2
  gen_name=$3

  echo "----------------------------------------"
  echo "Dataset: $dataset"
  echo "Run: $run_name"
  echo "Gen: $gen_name"
  echo "----------------------------------------"

  python evaluate.py \
    --dataset "$dataset" \
    --run_name "$run_name" \
    --gen_name "$gen_name"
}

CONFIGS=(
  "temp0.5_toppnone_topknone_pen0.0"
  "temp0.7_toppnone_topknone_pen0.0"

  "temp1.0_toppnone_topknone_pen0.0"
  "temp1.0_toppnone_topknone_pen0.1"
  "temp1.0_toppnone_topknone_pen0.3"

  "temp1.2_toppnone_topknone_pen0.0"
  "temp1.2_toppnone_topknone_pen0.1"
  "temp1.2_toppnone_topknone_pen0.3"

  "temp1.0_topp0.9_topknone_pen0.0"
  "temp1.0_topp0.95_topknone_pen0.0"

  "temp1.0_toppnone_topk10_pen0.0"
  "temp1.0_toppnone_topk50_pen0.0"

  "temp1.5_topp0.9_topknone_pen0.0"
  "temp1.5_topp0.9_topknone_pen0.1"
  "temp1.5_topp0.9_topknone_pen0.3"

  "temp1.5_topp0.95_topknone_pen0.0"
  "temp1.5_topp0.95_topknone_pen0.1"
  "temp1.5_topp0.95_topknone_pen0.3"

  "temp1.5_toppnone_topknone_pen0.0"
  "temp1.5_toppnone_topknone_pen0.1"
  "temp1.5_toppnone_topknone_pen0.3"
)

# ------------------------
# DATASETS
# ------------------------

DATASETS=(
  "movielens|final"
  "amazon_music|maxlen50_final"
  "amazon_games|maxlen50_final"
  "amazon_cds|maxlen50_final"
)

# ------------------------
# RUN
# ------------------------

for ds in "${DATASETS[@]}"; do
  IFS="|" read -r DATASET RUN_NAME <<< "$ds"

  for cfg in "${CONFIGS[@]}"; do
    run_eval "$DATASET" "$RUN_NAME" "$cfg"
  done
done

echo "=== Done ==="