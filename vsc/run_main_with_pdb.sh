# Change directory to location from which task was submitted to queue: $VSC_DATA/...
cd $PBS_O_WORKDIR
# Purge all existing software packages
module purge

# Load VSC optimized PyTorch with CUDA
module load PyTorch/1.0.1-intel-2018a

(>&1 echo "$@")
(>&1 echo 'starting computation')
# Execute the script with post-mortem debugger
python -m pdb ../main.py train --epochs 20 --batch_size 32 --data '/scratch/leuven/341/vsc34153/rt_st_gcn/data/pku-mmd-xsubject' --out '/scratch/leuven/341/vsc34153/rt_st_gcn/pretrained_models/pku-mmd-xsubject' --config '/data/leuven/341/vsc34153/rt-st-gcn/config/pku-mmd/realtime_vsc.json'
