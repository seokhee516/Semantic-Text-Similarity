#!/bin/bash
CONFIGS=("base_config" "electra_config")

config_length=${#CONFIGS[@]}


for (( i=0; i<${config_length}; i++ ))
do

    echo ${CONFIGS[$i]}
    python3 train_y.py \
        --config ${CONFIGS[$i]}
done