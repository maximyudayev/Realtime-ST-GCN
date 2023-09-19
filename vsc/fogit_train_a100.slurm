#!/bin/bash -l
#SBATCH -A lp_stadius_fpga_ai
#SBATCH --cluster=wice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=7000
#SBATCH --output=debug/%x_%a.out
#SBATCH --error=debug/%x_%a.err

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=maxim.yudayev@kuleuven.be

# (1/4) Full Icelake x4 A100 GPU node (requires 18 cores per gpu)
# GPU partition (3 day max walltime)
# Icelake GPU nodes have 512GB

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
mkdir -p ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD/${SLURM_ARRAY_TASK_ID}
cp -r ${VSC_DATA}/rt-st-gcn/data/imu_fogit_ABCD/* ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD/${SLURM_ARRAY_TASK_ID}
mv ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD/${SLURM_ARRAY_TASK_ID}/train/features/$(printf '%0.3d' ${SLURM_ARRAY_TASK_ID})* ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD/${SLURM_ARRAY_TASK_ID}/val/features
mv ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD/${SLURM_ARRAY_TASK_ID}/train/labels/$(printf '%0.3d' ${SLURM_ARRAY_TASK_ID})* ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD/${SLURM_ARRAY_TASK_ID}/val/labels

source activate rt-st-gcn

(>&1 echo "$@")
(>&1 echo 'starting computation')
# Execute and time the script
time python ../main.py train \
    --batch_size "$(ls ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD/${SLURM_ARRAY_TASK_ID}/val/labels | wc -l)" \
    --data "${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD/${SLURM_ARRAY_TASK_ID}" \
    "$@"

rm -r ${VSC_SCRATCH}/rt-st-gcn/data/imu_fogit_ABCD/${SLURM_ARRAY_TASK_ID}
