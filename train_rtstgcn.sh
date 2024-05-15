#!/bin/bash

CONFIG_FILE="config/pku-mmd/ln/shiftgcn_local.json"
BASE_OUT_DIR="pretrained_models/shiftgcn"

KERNELS=(9 21 69)

for KERNEL in "${KERNELS[@]}"
do
    jq ".arch.kernel = $KERNEL | .arch[\"shift-gcn\"].kernel = $KERNEL" $CONFIG_FILE > "$CONFIG_FILE.bak" && mv "$CONFIG_FILE.bak" $CONFIG_FILE

    NEW_OUT_DIR="${BASE_OUT_DIR}/gamma_${KERNEL}"
    jq ".processor.out = \"$NEW_OUT_DIR\"" $CONFIG_FILE > "$CONFIG_FILE.bak" && mv "$CONFIG_FILE.bak" $CONFIG_FILE

    python main.py train --config $CONFIG_FILE
done
