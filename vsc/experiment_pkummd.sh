#!/bin/bash
# sbatch --parsable --time=40:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_stgcn_asis" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/stgcn_vsc.json'
# sbatch --parsable --time=72:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_aagcn_asis" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/aagcn_vsc.json'
# sbatch --parsable --time=02:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_mstcn_asis" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/mstcn_vsc.json'
# sbatch --parsable --time=42:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_msgcn_asis" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/msgcn_vsc.json'
# sbatch --parsable --time=24:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_rtstgcn_asis" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/as_is/rtstgcn_vsc.json'

# sbatch --parsable --time=72:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_aagcn_softmax" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/softmax/aagcn_vsc.json'
# sbatch --parsable --time=02:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_mstcn_softmax" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/softmax/mstcn_vsc.json'
# sbatch --parsable --time=42:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_msgcn_softmax" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/softmax/msgcn_vsc.json'

# sbatch --parsable --time=40:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_stgcn_nobn" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/no_bn/stgcn_vsc.json'
# sbatch --parsable --time=72:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_aagcn_nobn" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/no_bn/aagcn_vsc.json'
# sbatch --parsable --time=42:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_msgcn_nobn" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/no_bn/msgcn_vsc.json'
# sbatch --parsable --time=24:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_rtstgcn_nobn" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/no_bn/rtstgcn_vsc.json'

# sbatch --parsable --time=02:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_mstcn_logits" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/refine_logits/mstcn_vsc.json'
# sbatch --parsable --time=42:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_msgcn_logits" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/refine_logits/msgcn_vsc.json'
# sbatch --parsable --time=72:00:00 --job-name="train_$(date +'%d-%m-%y-%T')_pkummd_aagcn_logits" pkummd_p100.slurm --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/refine_logits/aagcn_vsc.json'
