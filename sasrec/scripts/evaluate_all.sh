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

# ------------------------
# Movielens (run_name = final)
# ------------------------
DATASET="movielens"
RUN_NAME="final"

run_eval $DATASET $RUN_NAME temp0.5_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp0.7_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.2_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_topp0.9_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_topp0.95_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topk10_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topk50_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topknone_pen0.1
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topknone_pen0.3

# ------------------------
# Amazon Music
# ------------------------
DATASET="amazon_music"
RUN_NAME="maxlen50_final"

run_eval $DATASET $RUN_NAME temp0.5_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp0.7_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.2_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_topp0.9_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_topp0.95_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topk10_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topk50_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topknone_pen0.1
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topknone_pen0.3

# ------------------------
# Amazon Games
# ------------------------
DATASET="amazon_games"
RUN_NAME="maxlen50_final"

run_eval $DATASET $RUN_NAME temp0.5_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp0.7_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.2_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_topp0.9_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_topp0.95_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topk10_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topk50_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topknone_pen0.1
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topknone_pen0.3

# ------------------------
# Amazon CDs
# ------------------------
DATASET="amazon_cds"
RUN_NAME="maxlen50_final"

run_eval $DATASET $RUN_NAME temp0.5_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp0.7_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.2_toppnone_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_topp0.9_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_topp0.95_topknone_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topk10_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topk50_pen0.0
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topknone_pen0.1
run_eval $DATASET $RUN_NAME temp1.0_toppnone_topknone_pen0.3

echo "=== Done ==="