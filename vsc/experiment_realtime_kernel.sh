#!/bin/bash
# 1 epoch jobs to estimate the time needed per configuration
#qsub -l walltime=30:00 -N st_gcn_pkummd_rt_9_64_1_debug -F "--epochs 1 --kernel 9 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
#qsub -l walltime=30:00 -N st_gcn_pkummd_rt_21_64_1_debug -F "--epochs 1 --kernel 21 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
#qsub -l walltime=30:00 -N st_gcn_pkummd_rt_69_64_1_debug -F "--epochs 1 --kernel 69 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
#qsub -l walltime=30:00 -N st_gcn_pkummd_rt_153_64_1_debug -F "--epochs 1 --kernel 153 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
#qsub -l walltime=30:00 -N st_gcn_pkummd_rt_299_64_1_debug -F "--epochs 1 --kernel 299 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs


# Actual jobs
# Can all be run on P100 GPUs (Genius)
qsub -l walltime=08:00:00 -N st_gcn_pkummd_rt_9_64_50 -F "--epochs 50 --kernel 9 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
#qsub -l walltime=00:00:00 -N st_gcn_pkummd_rt_21_64_50 -F "--epochs 50 --kernel 21 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
#qsub -l walltime=00:00:00 -N st_gcn_pkummd_rt_69_64_50 -F "--epochs 50 --kernel 69 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
#qsub -l walltime=00:00:00 -N st_gcn_pkummd_rt_153_64_50 -F "--epochs 50 --kernel 153 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
#qsub -l walltime=00:00:00 -N st_gcn_pkummd_rt_299_64_50 -F "--epochs 50 --kernel 299 --batch_size 64 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=08:00:00 -N st_gcn_pkummd_sub_rt9_64_50 -F "--epochs 50 --kernel 9 --batch_size 64 --data '/scratch/leuven/341/vsc34153/rt-st-gcn/data/pku-mmd-xsubject' --out '/scratch/leuven/341/vsc34153/rt-st-gcn/pretrained_models/pku-mmd-xsubject' --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
#qsub -l walltime=00:00:00 -N st_gcn_pkummd_sub_rt21_64_50 -F "--epochs 50 --kernel 21 --batch_size 64 --data '/scratch/leuven/341/vsc34153/rt-st-gcn/data/pku-mmd-xsubject' --out '/scratch/leuven/341/vsc34153/rt-st-gcn/pretrained_models/pku-mmd-xsubject' --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
#qsub -l walltime=00:00:00 -N st_gcn_pkummd_sub_rt69_64_50 -F "--epochs 50 --kernel 69 --batch_size 64 --data '/scratch/leuven/341/vsc34153/rt-st-gcn/data/pku-mmd-xsubject' --out '/scratch/leuven/341/vsc34153/rt-st-gcn/pretrained_models/pku-mmd-xsubject' --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
#qsub -l walltime=00:00:00 -N st_gcn_pkummd_sub_rt153_64_50 -F "--epochs 50 --kernel 153 --batch_size 64 --data '/scratch/leuven/341/vsc34153/rt-st-gcn/data/pku-mmd-xsubject' --out '/scratch/leuven/341/vsc34153/rt-st-gcn/pretrained_models/pku-mmd-xsubject' --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
#qsub -l walltime=00:00:00 -N st_gcn_pkummd_sub_rt299_64_50 -F "--epochs 50 --kernel 299 --batch_size 64 --data '/scratch/leuven/341/vsc34153/rt-st-gcn/data/pku-mmd-xsubject' --out '/scratch/leuven/341/vsc34153/rt-st-gcn/pretrained_models/pku-mmd-xsubject' --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs
