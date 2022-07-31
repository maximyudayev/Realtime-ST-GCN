#!/bin/bash
source activate ml 2> /dev/null
if [ $? -ne 0 ]
then
    (>&2 echo '### [ERR]: conda environment not sourced successfully (.pbs)')
fi

(>&2 echo 'starting computation')
# Execute and time the script
time python3 ../main.py train
