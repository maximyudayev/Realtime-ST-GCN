#!/bin/bash
#SBATCH -A lp_stadius_fpga_ai
#SBATCH --cluster=genius
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu_p100
#SBATCH --mem-per-cpu=5000

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=maxim.yudayev@kuleuven.be

# (1/20) Full Skylake x4 P100 GPU node (requires 9 cores per gpu)
# GPU partition
# Skylake GPU nodes have 192GB

mail -s "[$SLURM_JOB_NAME]: STARTED" maxim.yudayev@kuleuven.be <<< ""

# Change directory to location from which task was submitted to queue: $VSC_DATA/...
cd $SLURM_SUBMIT_DIR
# Purge all existing software packages
module purge

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate rt-st-gcn

(>&1 echo "$@")
(>&1 echo 'starting computation')
# Execute and time the script
time python ../main.py train "$@"
