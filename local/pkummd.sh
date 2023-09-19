#!/bin/bash -l
#SBATCH -A lp_stadius_fpga_ai
#SBATCH --cluster=genius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu_p100_debug
#SBATCH --mem-per-cpu=5g
#SBATCH --output=debug/%x.out
#SBATCH --error=debug/%x.err

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=maxim.yudayev@kuleuven.be

# (1/1) Debug Full Skylake x4 P100 GPU node (requires 9 cores per gpu)
# GPU partition (gpu_p100_debug - 30 minute max walltime)
# Skylake GPU nodes have 192GB
# Debug specifier
# Debug jobs limited to 30 min (max 1 debug job in the queue)

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
