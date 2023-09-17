#!/bin/bash
# Debug job
sbatch --parsable --time=00:10:00 --job-name="debug_$(date +'%d-%m-%y-%T')_pkummd_rtstgcn_9_1" fogit_debug_p100.slurm --epochs 1 --kernel 9 --segment 500 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'

# Actual jobs
# sbatch --parsable --time=00:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_rtstgcn_9_50" fogit_train_p100.slurm --epochs 50 --kernel 9 --segment 500 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'
