#!/bin/bash
qsub -l walltime=2:00:00 -N st_gcn_pkummd_rt_9_32_20 -F "--epochs 20 --batch_size 32 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=3:00:00 -N st_gcn_pkummd_adapt_9_32_20 -F "--epochs 20 --batch_size 32 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/adapted_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=10:00:00 -N st_gcn_pkummd_orig_9_8_20 -F "--epochs 20 --batch_size 8 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=3:00:00 -N st_gcn_pkummd_orig_red_9_32_20 -F "--epochs 20 --batch_size 32 --latency --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'" st_gcn_gpu_train.pbs


#qsub -l walltime=2:00:00 -N st_gcn_pkummd_sub_rt9_20 -F "--epochs 20 --batch_size 32 --data '/scratch/leuven/341/vsc34153/rt-st-gcn/data/pku-mmd-xsubject' --out '/scratch/leuven/341/vsc34153/rt-st-gcn/pretrained_models/pku-mmd-xsubject' --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

#qsub -l walltime=3:00:00 -N st_gcn_pkummd_sub_adapt9_20 -F "--epochs 20 --batch_size 32 --data '/scratch/leuven/341/vsc34153/rt-st-gcn/data/pku-mmd-xsubject' --out '/scratch/leuven/341/vsc34153/rt-st-gcn/pretrained_models/pku-mmd-xsubject' --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/adapted_vsc.json'" st_gcn_gpu_train.pbs

#qsub -l walltime=10:00:00 -N st_gcn_pkummd_sub_orig9_20 -F "--epochs 20 --batch_size 8 --data '/scratch/leuven/341/vsc34153/rt-st-gcn/data/pku-mmd-xsubject' --out '/scratch/leuven/341/vsc34153/rt-st-gcn/pretrained_models/pku-mmd-xsubject' --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'" st_gcn_gpu_train.pbs

#qsub -l walltime=3:00:00 -N st_gcn_pkummd_sub_orig_red9_20 -F "--epochs 20 --batch_size 32 --latency --data '/scratch/leuven/341/vsc34153/rt-st-gcn/data/pku-mmd-xsubject' --out '/scratch/leuven/341/vsc34153/rt-st-gcn/pretrained_models/pku-mmd-xsubject' --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'" st_gcn_gpu_train.pbs


#qsub -l walltime=2:00:00 -N st_gcn_pkummd_rt21_20 -F "--kernel 21 --epochs 20 --batch_size 32 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

#qsub -l walltime=3:00:00 -N st_gcn_pkummd_adapt21_20 -F "--kernel 21 --epochs 20 --batch_size 32 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/adapted_vsc.json'" st_gcn_gpu_train.pbs

#qsub -l walltime=10:00:00 -N st_gcn_pkummd_orig21_20 -F "--kernel 21 --epochs 20 --batch_size 8 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'" st_gcn_gpu_train.pbs

#qsub -l walltime=3:00:00 -N st_gcn_pkummd_orig_red21_20 -F "--kernel 21 --epochs 20 --batch_size 32 --latency --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'" st_gcn_gpu_train.pbs
