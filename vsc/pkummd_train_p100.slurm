#!/bin/bash -l
#SBATCH -A lp_stadius_fpga_ai
#SBATCH --cluster=genius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu_p100
#SBATCH --mem-per-cpu=5000
#SBATCH --output=debug/%x.out
#SBATCH --error=debug/%x.err

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=maxim.yudayev@kuleuven.be

# (1/20) Full Skylake x4 P100 GPU node (requires 9 cores per gpu)
# GPU partition (gpu_p100 - 3 day max walltime; gpu_p100_long - 7 day max walltime)
# Skylake GPU nodes have 192GB

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

export MASTER_PORT="$port"

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
