#!/bin/bash
# Train debug job
#sbatch --parsable --array=[13] --time=00:02:00 --job-name="debug_$(date +'%d-%m-%y-%T')_fogit_rtstgcn_9_1" fogit_p100_debug_copy.slurm --epochs 1 --kernel 9 --segment 1000 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/imu_fogit_ABCD/realtime_vsc.json'

# Test debug job
#sbatch --parsable --array=[13] --time=00:02:00 --job-name="debug_$(date +'%d-%m-%y-%T')_fogit_rtstgcn_9_1" fogit_p100_debug_copy.slurm --kernel 9 --segment 1000 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/imu_fogit_ABCD/realtime_vsc.json'

# Train jobs
#sbatch --parsable --array=[1-13] --time=00:30:00 --job-name="train_$(date +'%d-%m-%y-%T')_fogit_rtstgcn_9_30" fogit_p100.slurm --epochs 30 --kernel 9 --segment 1000 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/imu_fogit_ABCD/realtime_vsc.json'

# Test jobs
sbatch --parsable --array=[1-13] --time=00:02:00 --job-name="test_$(date +'%d-%m-%y-%T')_fogit_rtstgcn_9" fogit_p100.slurm --kernel 9 --segment 1000 --config '/data/leuven/341/vsc34153/rt-st-gcn/config/imu_fogit_ABCD/realtime_vsc.json'

