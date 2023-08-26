#!/bin/bash

link=2
num_nb=4
sigma=0.0
alpha=1.3
tau_smooth=0.6
cost_name="MC-CNN"
use_cost_vol=1
min_level=2

DATASETS=("Adirondack" "ArtL" "Jadeplant" "Motorcycle" "MotorcycleE" "Piano" "PianoL" "Pipes" "Playroom" "Playtable" "PlaytableP" "Recycle" "Shelves" "Teddy" "Vintage")

mkdir -p results/trainingH

for DATASET in "${DATASETS[@]}"
do
    ./build/bench_middv3 data/trainingH/$DATASET/ results/trainingH/$DATASET/ ${link} ${num_nb} ${sigma} ${alpha} ${tau_smooth} ${cost_name} ${use_cost_vol} ${min_level}
done