#!/bin/bash
qsub -l walltime=24:00:00 -N st_gcn_pkummd_rt9_30 -F "--epochs 30 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=24:00:00 -N st_gcn_pkummd_rt21_30 -F "--kernel 21 --epochs 30 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=24:00:00 -N st_gcn_pkummd_rt69_30 -F "--kernel 69 --epochs 30 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=24:00:00 -N st_gcn_pkummd_rt153_30 -F "--kernel 153 --epochs 30 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=24:00:00 -N st_gcn_pkummd_rt299_30 -F "--kernel 299 --epochs 30 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=24:00:00 -N st_gcn_pkummd_orig9_30 -F "--epochs 30 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=24:00:00 -N st_gcn_pkummd_orig21_30 -F "--kernel 21 --epochs 30 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'" st_gcn_gpu_train.pbs
