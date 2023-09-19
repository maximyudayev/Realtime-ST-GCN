#!/bin/bash -l
mv data/imu_fogit_ABCD/val/features/* data/imu_fogit_ABCD/train/features
mv data/imu_fogit_ABCD/val/labels/* data/imu_fogit_ABCD/train/labels

export SLURM_JOB_NAME=
export SLURM_ARRAY_TASK_ID=
