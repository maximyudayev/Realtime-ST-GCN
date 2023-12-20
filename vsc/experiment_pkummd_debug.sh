#!/bin/bash
TIMESTAMP="$(date +'%d-%m-%y-%T')"

#sbatch --parsable --time=00:30:00 --job-name="debug_${TIMESTAMP}_pkummd_stgcn" pkummd_a100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/stgcn_vsc.json'
#sbatch --parsable --time=00:30:00 --job-name="debug_${TIMESTAMP}_pkummd_aagcn" pkummd_a100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/aagcn_vsc.json'
#sbatch --parsable --time=00:10:00 --job-name="debug_${TIMESTAMP}_pkummd_mstcn" pkummd_p100_debug.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/mstcn_vsc.json'
#sbatch --parsable --time=00:30:00 --job-name="debug_${TIMESTAMP}_pkummd_msgcn" pkummd_a100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/msgcn_vsc.json'
#sbatch --parsable --time=00:10:00 --job-name="debug_${TIMESTAMP}_pkummd_rtstgcn" pkummd_p100_debug.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/rtstgcn_vsc.json'
