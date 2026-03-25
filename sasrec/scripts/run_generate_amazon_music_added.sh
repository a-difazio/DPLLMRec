#!/bin/bash
# run_generate_amazon_music.sh

DEVICE=${1:-cuda}  # default cuda, override con: bash script.sh mps

echo '=== Generating for amazon_music (device: $DEVICE) ==='

echo 'Running temp1.2_toppnone_topknone_pen0.1'
python generate.py \
    --dataset amazon_music \
    --run_name maxlen50_final \
    --gen_name temp1.2_toppnone_topknone_pen0.1 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.2 \
    --penalty 0.1 \
    --seed 42

echo 'Running temp1.2_toppnone_topknone_pen0.3'
python generate.py \
    --dataset amazon_music \
    --run_name maxlen50_final \
    --gen_name temp1.2_toppnone_topknone_pen0.3 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.2 \
    --penalty 0.3 \
    --seed 42

echo 'Running temp1.5_toppnone_topknone_pen0.0'
python generate.py \
    --dataset amazon_music \
    --run_name maxlen50_final \
    --gen_name temp1.5_toppnone_topknone_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.5 \
    --penalty 0.0 \
    --seed 42

echo 'Running temp1.5_toppnone_topknone_pen0.1'
python generate.py \
    --dataset amazon_music \
    --run_name maxlen50_final \
    --gen_name temp1.5_toppnone_topknone_pen0.1 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.5 \
    --penalty 0.1 \
    --seed 42

echo 'Running temp1.5_toppnone_topknone_pen0.3'
python generate.py \
    --dataset amazon_music \
    --run_name maxlen50_final \
    --gen_name temp1.5_toppnone_topknone_pen0.3 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.5 \
    --penalty 0.3 \
    --seed 42

echo 'Running temp1.5_topp0.9_topknone_pen0.0'
python generate.py \
    --dataset amazon_music \
    --run_name maxlen50_final \
    --gen_name temp1.5_topp0.9_topknone_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.5 \
    --penalty 0.0 \
    --seed 42 \
    --top_p 0.9

echo 'Running temp1.5_topp0.95_topknone_pen0.0'
python generate.py \
    --dataset amazon_music \
    --run_name maxlen50_final \
    --gen_name temp1.5_topp0.95_topknone_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.5 \
    --penalty 0.0 \
    --seed 42 \
    --top_p 0.95

echo 'Running temp1.5_topp0.9_topknone_pen0.1'
python generate.py \
    --dataset amazon_music \
    --run_name maxlen50_final \
    --gen_name temp1.5_topp0.9_topknone_pen0.1 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.5 \
    --penalty 0.1 \
    --seed 42 \
    --top_p 0.9

echo 'Running temp1.5_topp0.95_topknone_pen0.1'
python generate.py \
    --dataset amazon_music \
    --run_name maxlen50_final \
    --gen_name temp1.5_topp0.95_topknone_pen0.1 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.5 \
    --penalty 0.1 \
    --seed 42 \
    --top_p 0.95

echo 'Running temp1.5_topp0.9_topknone_pen0.3'
python generate.py \
    --dataset amazon_music \
    --run_name maxlen50_final \
    --gen_name temp1.5_topp0.9_topknone_pen0.3 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.5 \
    --penalty 0.3 \
    --seed 42 \
    --top_p 0.9

echo 'Running temp1.5_topp0.95_topknone_pen0.3'
python generate.py \
    --dataset amazon_music \
    --run_name maxlen50_final \
    --gen_name temp1.5_topp0.95_topknone_pen0.3 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.5 \
    --penalty 0.3 \
    --seed 42 \
    --top_p 0.95

echo '=== Done amazon_music ==='