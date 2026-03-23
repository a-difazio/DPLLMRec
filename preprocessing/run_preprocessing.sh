#!/bin/bash
set -e

datasets=(
  movielens
  amazon_music
  amazon_cds
  amazon_games
)

for d in "${datasets[@]}"; do
  echo "Processing $d..."
  python preprocess.py -d "$d"
done