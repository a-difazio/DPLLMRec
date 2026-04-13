#!/bin/bash
# run_generate_amazon_music_tsts.sh

DEVICE=${1:-cuda}  # default cuda, override con: bash script.sh mps

echo '=== Generating for amazon_music_tsts (device: $DEVICE) ==='

echo 'Running temp1.0_toppnone_topknone_pen0.3'
python generate.py \
    --dataset amazon_music_tsts \
    --run_name maxlen50_final \
    --gen_name temp1.0_toppnone_topknone_pen0.3 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.0 \
    --penalty 0.3 \
    --seed 42

echo 'Running temp1.0_topp0.9_topknone_pen0.0'
python generate.py \
    --dataset amazon_music_tsts \
    --run_name maxlen50_final \
    --gen_name temp1.0_topp0.9_topknone_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.0 \
    --penalty 0.0 \
    --seed 42 \
    --top_p 0.9

echo 'Running temp1.2_toppnone_topknone_pen0.0'
python generate.py \
    --dataset amazon_music_tsts \
    --run_name maxlen50_final \
    --gen_name temp1.2_toppnone_topknone_pen0.0 \
    --checkpoint checkpoint_best.pth \
    --device $DEVICE \
    --temperature 1.2 \
    --penalty 0.0 \
    --seed 42

echo '=== Done amazon_music_tsts ==='