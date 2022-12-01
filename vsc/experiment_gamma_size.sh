#!/bin/bash
qsub -l walltime=6:00:00 -N st_gcn_pkummd_rt9_100 -F "--config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=6:00:00 -N st_gcn_pkummd_rt21_100 -F "--kernel 21 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=6:00:00 -N st_gcn_pkummd_rt69_100 -F "--kernel 69 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=6:00:00 -N st_gcn_pkummd_rt153_100 -F "--kernel 153 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=8:00:00 -N st_gcn_pkummd_rt299_100 -F "--kernel 299 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=6:00:00 -N st_gcn_pkummd_orig9_100 -F "--config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=9:00:00 -N st_gcn_pkummd_orig21_100 -F "--kernel 21 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'" st_gcn_gpu_train.pbs

qsub -l walltime=15:00:00 -N st_gcn_pkummd_orig69_100 -F "--kernel 69 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc.json'" st_gcn_gpu_train.pbs


qsub -l walltime=6:00:00 -N st_gcn_pkummd_rt9_100_do -F "--config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc_do.json'" st_gcn_gpu_train.pbs

qsub -l walltime=6:00:00 -N st_gcn_pkummd_rt21_100_do -F "--kernel 21 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc_do.json'" st_gcn_gpu_train.pbs

qsub -l walltime=6:00:00 -N st_gcn_pkummd_rt69_100_do -F "--kernel 69 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc_do.json'" st_gcn_gpu_train.pbs

qsub -l walltime=6:00:00 -N st_gcn_pkummd_rt153_100_do -F "--kernel 153 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc_do.json'" st_gcn_gpu_train.pbs

qsub -l walltime=8:00:00 -N st_gcn_pkummd_rt299_100_do -F "--kernel 299 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc_do.json'" st_gcn_gpu_train.pbs

qsub -l walltime=6:00:00 -N st_gcn_pkummd_orig9_100_do -F "--config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc_do.json'" st_gcn_gpu_train.pbs

qsub -l walltime=9:00:00 -N st_gcn_pkummd_orig21_100_do -F "--kernel 21 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc_do.json'" st_gcn_gpu_train.pbs

qsub -l walltime=15:00:00 -N st_gcn_pkummd_orig69_100_do -F "--kernel 69 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/original_vsc_do.json'" st_gcn_gpu_train.pbs
