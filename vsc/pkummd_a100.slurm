#!/bin/bash -l
#SBATCH -A lp_stadius_fpga_ai
#SBATCH --cluster=wice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=7000
#SBATCH --output=debug/%x.out
#SBATCH --error=debug/%x.err

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=maxim.yudayev@kuleuven.be

# (1/4) Full Icelake x4 A100 GPU node (requires 18 cores per gpu)
# GPU partition (3 day max walltime)
# Icelake GPU nodes have 512GB

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
