#!/bin/bash -l
#SBATCH -A lp_stadius_fpga_ai
#SBATCH --cluster=genius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_p100_debug
#SBATCH --mem-per-cpu=5000
#SBATCH --output=debug/%x_%a.out
#SBATCH --error=debug/%x_%a.err

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=maxim.yudayev@kuleuven.be

# (1/1) Debug Full Skylake x4 P100 GPU node (requires 9 cores per gpu)
# GPU partition (gpu_p100_debug - 30 minute max walltime)
# Skylake GPU nodes have 192GB
# Debug specifier
# Debug jobs limited to 30 min (max 1 debug job in the queue)

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

export MASTER_PORT="$port"

if ["$SLURM_ARRAY_TASK_ID" == ""]
then
    JOB="${SLURM_JOB_NAME}"
else
    JOB="${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}"
fi

mail -s "[${JOB}]: STARTED" maxim.yudayev@kuleuven.be <<< ""

# Change directory to location from which task was submitted to queue: $VSC_DATA/...
cd $SLURM_SUBMIT_DIR
# Purge all existing software packages
module purge

# Create temporary data directories on the Scratch partition and copy the data over for leave-one-subject-out training protocol
mkdir -p ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD_copy/${SLURM_ARRAY_TASK_ID}
cp -r ${VSC_DATA}/rt-st-gcn/data/imu_fogit_ABCD/* ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD_copy/${SLURM_ARRAY_TASK_ID}
mv ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD_copy/${SLURM_ARRAY_TASK_ID}/train/features/$(printf '%0.3d' ${SLURM_ARRAY_TASK_ID})* ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD_copy/${SLURM_ARRAY_TASK_ID}/val/features
mv ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD_copy/${SLURM_ARRAY_TASK_ID}/train/labels/$(printf '%0.3d' ${SLURM_ARRAY_TASK_ID})* ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD_copy/${SLURM_ARRAY_TASK_ID}/val/labels

source activate rt-st-gcn

(>&1 echo "$@")
(>&1 echo 'starting computation')
# Execute and time the script
time python ../main.py train \
    --batch_size "$(ls ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD_copy/${SLURM_ARRAY_TASK_ID}/train/labels | wc -l)" \
    --demo $(shuf -i 0-$(($(ls ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD_copy/${SLURM_ARRAY_TASK_ID}/val/labels | wc -l)-1)) -n 3) \
    --data "${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD_copy/${SLURM_ARRAY_TASK_ID}" \
    "$@"
#    --checkpoint "${VSC_SCRATCH}/rt-st-gcn/pretrained_models/imu_fogit_ABCD/train_27-09-23-04:44:30_fogit_stgcn_9_300_10_${SLURM_ARRAY_TASK_ID}/final.pt" \

rm -r ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD_copy/${SLURM_ARRAY_TASK_ID}
