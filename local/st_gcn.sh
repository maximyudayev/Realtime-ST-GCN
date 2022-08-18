#!/bin/bash
conda activate ml 2> /dev/null
if [ $? -ne 0 ]
then
    (>&2 echo '### [ERR]: conda environment not sourced successfully (.pbs)')
fi

(>&2 echo 'starting computation')
# Execute and time the script
# time python3 ../main.py train

# Execute and profile the script
python3 -m cProfile -s time -o st-gcn-proposed.prof main.py train
python3 -m cProfile -s time -o st-gcn-original.prof main.py train --config config/kinetics/config_original_local.json
