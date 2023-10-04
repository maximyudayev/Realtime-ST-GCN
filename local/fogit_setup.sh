#!/bin/bash -l
export SLURM_JOB_NAME="debug_$(date +'%d-%m-%y-%T')_fogit_rtstgcn_${2}_${3}"
# export SLURM_JOB_NAME="debug_$(date +'%d-%m-%y-%T')_fogit_stgcn_${2}_${3}_${4}"
export SLURM_ARRAY_TASK_ID=${1}
export MASTER_ADDR="localhost"
export MASTER_PORT=42069

mv data/imu_fogit_ABCD/train/features/$(printf '%0.3d' ${1})* data/imu_fogit_ABCD/val/features
mv data/imu_fogit_ABCD/train/labels/$(printf '%0.3d' ${1})* data/imu_fogit_ABCD/val/labels
echo $(ls data/imu_fogit_ABCD/train/labels | wc -l)
