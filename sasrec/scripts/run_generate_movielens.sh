#!/bin/bash
# run_generate_movielens.sh

DEVICE=${1:-cuda}  # default cuda, override con: bash script.sh mps

echo '=== Generating for movielens (device: $DEVICE) ==='

echo 'Running temp0.5_toppnone_topknone_pen0.0'
python generate.py \
    --dataset movielens \
    --run_name final \
    --gen_name temp0.5_toppnone_topknone_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 0.5 \
    --penalty 0.0 \
    --seed 42

echo 'Running temp0.7_toppnone_topknone_pen0.0'
python generate.py \
    --dataset movielens \
    --run_name final \
    --gen_name temp0.7_toppnone_topknone_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 0.7 \
    --penalty 0.0 \
    --seed 42

echo 'Running temp1.0_toppnone_topknone_pen0.0'
python generate.py \
    --dataset movielens \
    --run_name final \
    --gen_name temp1.0_toppnone_topknone_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.0 \
    --penalty 0.0 \
    --seed 42

echo 'Running temp1.2_toppnone_topknone_pen0.0'
python generate.py \
    --dataset movielens \
    --run_name final \
    --gen_name temp1.2_toppnone_topknone_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.2 \
    --penalty 0.0 \
    --seed 42

echo 'Running temp1.0_topp0.9_topknone_pen0.0'
python generate.py \
    --dataset movielens \
    --run_name final \
    --gen_name temp1.0_topp0.9_topknone_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.0 \
    --penalty 0.0 \
    --seed 42 \
    --top_p 0.9

echo 'Running temp1.0_topp0.95_topknone_pen0.0'
python generate.py \
    --dataset movielens \
    --run_name final \
    --gen_name temp1.0_topp0.95_topknone_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.0 \
    --penalty 0.0 \
    --seed 42 \
    --top_p 0.95

echo 'Running temp1.0_toppnone_topk10_pen0.0'
python generate.py \
    --dataset movielens \
    --run_name final \
    --gen_name temp1.0_toppnone_topk10_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.0 \
    --penalty 0.0 \
    --seed 42 \
    --top_k 10

echo 'Running temp1.0_toppnone_topk50_pen0.0'
python generate.py \
    --dataset movielens \
    --run_name final \
    --gen_name temp1.0_toppnone_topk50_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.0 \
    --penalty 0.0 \
    --seed 42 \
    --top_k 50

echo 'Running temp1.0_toppnone_topknone_pen0.1'
python generate.py \
    --dataset movielens \
    --run_name final \
    --gen_name temp1.0_toppnone_topknone_pen0.1 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.0 \
    --penalty 0.1 \
    --seed 42

echo 'Running temp1.0_toppnone_topknone_pen0.3'
python generate.py \
    --dataset movielens \
    --run_name final \
    --gen_name temp1.0_toppnone_topknone_pen0.3 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.0 \
    --penalty 0.3 \
    --seed 42

echo '=== Done movielens ==='