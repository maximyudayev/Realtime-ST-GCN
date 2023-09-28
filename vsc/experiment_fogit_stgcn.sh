#!/bin/bash
# Debug job
#sbatch --parsable --array=[13] --time=00:30:00 --job-name="debug_$(date +'%d-%m-%y-%T')_fogit_stgcn_9_300_2" fogit_debug_p100.slurm --epochs 2 --kernel 9 --receptive_field 300 --segment 350 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/imu_fogit_ABCD/original_vsc.json'

# Actual jobs
sbatch --parsable --array=[1-13] --time=06:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_fogit_stgcn_9_300_30" fogit_train_p100.slurm --epochs 30 --kernel 9 --receptive_field 300 --segment 350 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/imu_fogit_ABCD/original_vsc.json'
