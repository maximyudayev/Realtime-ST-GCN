#!/bin/bash
TIMESTAMP=$(date +'%d-%m-%y-%T')

sbatch --parsable --time=40:00:00 --job-name="train_${TIMESTAMP}_pkummd_stgcn_asis" --ntasks-per-node=36 --gpus-per-node=2 pkummd_a100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/stgcn_vsc.json'
sbatch --parsable --time=72:00:00 --job-name="train_${TIMESTAMP}_pkummd_aagcn_asis" --ntasks-per-node=72 --gpus-per-node=4 pkummd_a100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/aagcn_vsc.json'
sbatch --parsable --time=60:00:00 --job-name="train_${TIMESTAMP}_pkummd_msgcn_asis" --ntasks-per-node=54 --gpus-per-node=3 pkummd_a100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/msgcn_vsc.json'
#sbatch --parsable --time=20:00:00 --job-name="train_${TIMESTAMP}_pkummd_mstcn_asis" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/mstcn_vsc.json'
#sbatch --parsable --time=6:00:00 --job-name="train_${TIMESTAMP}_pkummd_rtstgcn_bn" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/rtstgcn_vsc.json'

sbatch --parsable --time=40:00:00 --job-name="train_${TIMESTAMP}_pkummd_stgcn_ln" --ntasks-per-node=36 --gpus-per-node=2 pkummd_a100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/ln/stgcn_vsc.json'
sbatch --parsable --time=72:00:00 --job-name="train_${TIMESTAMP}_pkummd_aagcn_ln" --ntasks-per-node=72 --gpus-per-node=4 pkummd_a100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/ln/aagcn_vsc.json'
sbatch --parsable --time=60:00:00 --job-name="train_${TIMESTAMP}_pkummd_msgcn_ln" --ntasks-per-node=54 --gpus-per-node=3 pkummd_a100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/ln/msgcn_vsc.json'
#sbatch --parsable --time=6:00:00 --job-name="train_${TIMESTAMP}_pkummd_rtstgcn_ln" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/ln/rtstgcn_vsc.json'
