#!/bin/bash
# Debug job
sbatch --parsable --time=00:10:00 --job-name="debug_$(date +'%d-%m-%y-%T')_pkummd_stgcn_9_50_2" pkummd_debug_p100.slurm --epochs 2 --kernel 9 --receptive_field 50 --segment 500 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'

# Actual jobs
# sbatch --parsable --time=00:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_stgcn_9_300_10" pkummd_train_p100.slurm --epochs 10 --kernel 9 --receptive_field 300 --segment 500 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'
