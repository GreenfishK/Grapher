#!/bin/bash

# 1 ... edge classifier head; 0 ... edge generation head
export MODEL_VARIANT=1

# Available hardware settings that are defined below
# No SLURM; one node; two GPUs per node; Nvidia A40: 1_2_a40
# No SLURM; one node;  two GPUs per node; Nvidia A100: 1_2_a100
# No SLURM; one node; two GPUs per node; Nvidia TI: 1_2_ti
# SLURM; two node; two GPUs per node; Nvidia A40: s2_2_a40
# SLURM; one node; two GPUs per node; Nvidia A40: s1_2_a40
# SLURM; one node; one GPU per node; Nvidia A40: s1_1_a40
export HARDWARE_SETTING=1_2_a40

# Load environment variables including HARDWARE_SETTING for the run
cd ../src 
source .env 

# Hyperparameters: Change them if needed. The env variables in the positional arguments are set in .env and set_hparams.sh
python_args=(
    "--run" "inference"
    "--edges_as_classes" "${MODEL_VARIANT}"
    "--dataset" "webnlg"
    "--checkpoint_model_id" "-2"
    "--default_root_dir" "${STORAGE_DRIVE}/data/core/grapher/output"
    "--inference_input_text" "'Danielle Harris had a main role in Super Capers, a 98 minute long movie.'"
)

# Check if $HARDWARE_SETTING starts with "s" for SLURM
if [[ $HARDWARE_SETTING == s* ]]; then
    # Run the inference with sbatch
    if [[ ${NUM_NODES} -gt 1 ]]; then
        sbatch --job-name=train_grapher_${MODEL_VARIANT}_${HARDWARE_SETTING} \
        --partition=${PARTITION} \
        --qos=${QOS} \
        --nodes=${NUM_NODES} \
        --gres=gpu:${NUM_GPUS_PER_NODE} \
        --ntasks-per-node=${NUM_GPUS_PER_NODE} \
        --time=1-0 \
        --mail-type=BEGIN \
        --mail-user=filip.kovacevic@tuwien.ac.at \
        main.slm "${python_args[@]}"
    else
        sbatch --job-name=train_grapher_${MODEL_VARIANT}_${HARDWARE_SETTING} \
        --partition=${PARTITION} \
        --qos=${QOS} \
        --gres=gpu:${NUM_GPUS_PER_NODE} \
        --ntasks-per-node=${NUM_GPUS_PER_NODE} \
        --time=1-0 \
        --mail-type=BEGIN \
        --mail-user=filip.kovacevic@tuwien.ac.at \
        main.slm "${python_args[@]}"
    fi
else
    # Run the inference without sbatch
    python3 main.py "${python_args[@]}"
fi