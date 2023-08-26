#!/bin/bash

output_dir=$1
algo_name=$2
thresh=$3

DATASETS=("Adirondack" "ArtL" "Jadeplant" "Motorcycle" "MotorcycleE" "Piano" "PianoL" "Pipes" "Playroom" "Playtable" "PlaytableP" "Recycle" "Shelves" "Teddy" "Vintage")

#mkdir -p results/trainingH

for DATASET in "${DATASETS[@]}"
do
    ./build/eval_middv3 data/trainingH/$DATASET/ $output_dir/trainingH/$DATASET/ ${thresh} ${algo_name}
done