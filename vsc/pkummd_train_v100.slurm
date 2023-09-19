#!/bin/bash -l
#SBATCH -A lp_stadius_fpga_ai
#SBATCH --cluster=genius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gpus-per-node=8
#SBATCH --partition=gpu_v100
#SBATCH --mem-per-cpu=20000
#SBATCH --output=debug/%x.out
#SBATCH --error=debug/%x.err

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=maxim.yudayev@kuleuven.be

# (1/2) Full Cascadelake x8 V100 GPU node (requires 4 cores per gpu)
# GPU partition (gpu_v100 - 3 day max walltime; gpu_v100_long - 7 day max walltime)
# Cascadelake GPU nodes have 768GB

mail -s "[${SLURM_JOB_NAME}]: STARTED" maxim.yudayev@kuleuven.be <<< ""

# Change directory to location from which task was submitted to queue: $VSC_DATA/...
cd $SLURM_SUBMIT_DIR
# Purge all existing software packages
module purge

source activate rt-st-gcn

(>&1 echo "$@")
(>&1 echo 'starting computation')
# Execute and time the script
time python ../main.py train "$@"
