#!/bin/bash
qsub -l walltime=20:00:00 -N st_gcn_train_rt21_40_epochs -F \
    "--kernel 21 \
    --epochs 40 \
    --config '/data/leuven/341/vsc34153/rt-st-gcn/config/kinetics/realtime_vsc.json'" \
    st_gcn_gpu_train.pbs

qsub -l walltime=10:00:00 -N st_gcn_train_rt69_40_epochs -F \
    "--kernel 69 \ 
    --epochs 40 \ 
    --checkpoint '/scratch/leuven/341/vsc34153/rt_st_gcn/pretrained_models/kinetics/realtime/run_51240758/epoch-19.pt' \
    --config '/data/leuven/341/vsc34153/rt-st-gcn/config/kinetics/realtime_vsc.json'" \
    st_gcn_gpu_train.pbs

qsub -l walltime=20:00:00 -N st_gcn_train_rt153_40_epochs -F \
    "--kernel 153 \
    --epochs 40 \
    --config '/data/leuven/341/vsc34153/rt-st-gcn/config/kinetics/realtime_vsc.json'" \
    st_gcn_gpu_train.pbs

qsub -l walltime=20:00:00 -N st_gcn_train_rt299_40_epochs -F \
    "--kernel 299 \
    --epochs 40 \
    --config '/data/leuven/341/vsc34153/rt-st-gcn/config/kinetics/realtime_vsc.json'" \
    st_gcn_gpu_train.pbs

qsub -l walltime=22:00:00 -N st_gcn_train_orig21_10_epochs -F \
    "--kernel 21 \
    --epochs 10 \
    --config '/data/leuven/341/vsc34153/rt-st-gcn/config/kinetics/original_vsc.json'" \
    st_gcn_gpu_train.pbs

qsub -l walltime=16:00:00 -N st_gcn_train_orig69_20_epochs -F \
    "--kernel 69 \
    --epochs 20 \
    --checkpoint '/scratch/leuven/341/vsc34153/rt_st_gcn/pretrained_models/kinetics/original/run_51240757/epoch-12.pt' \
    --config '/data/leuven/341/vsc34153/rt-st-gcn/config/kinetics/original_vsc.json'" \
    st_gcn_gpu_train.pbs

qsub -l walltime=22:00:00 -N st_gcn_train_orig153_10_epochs -F \
    "--kernel 153 \
    --epochs 10 \
    --config '/data/leuven/341/vsc34153/rt-st-gcn/config/kinetics/original_vsc.json'" \
    st_gcn_gpu_train.pbs

qsub -l walltime=22:00:00 -N st_gcn_train_orig299_10_epochs -F \
    "--kernel 299 \
    --epochs 10 \
    --config '/data/leuven/341/vsc34153/rt-st-gcn/config/kinetics/original_vsc.json'" \
    st_gcn_gpu_train.pbs
