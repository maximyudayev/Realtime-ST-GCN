#!/bin/bash
# Debug job
sbatch --parsable --array=[13] --time=00:10:00 --job-name="debug_$(date +'%d-%m-%y-%T')_fogit_rtstgcn_9_1" fogit_debug_p100_copy.slurm --epochs 1 --kernel 9 --segment 1000 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/imu_fogit_ABCD/realtime_vsc.json'

# Actual jobs
# sbatch --parsable --array=[1-13] --time=00:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_fogit_rtstgcn_9_30" fogit_train_p100.slurm --epochs 30 --kernel 9 --segment 1000 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/imu_fogit_ABCD/realtime_vsc.json'
