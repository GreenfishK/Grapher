#!/bin/bash

# Load environment variables including HARDWARE_SETTING for the run
cd ..
source .env 

# Hyperparameters: Change them if needed. The env variables in the positional arguments are set in .env and set_hparams.sh
python_args=(
    "--run" "inference"
    "--edges_as_classes" "${MODEL_VARIANT}"
    "--dataset" "webnlg"
    "--checkpoint_model_id" "${CHECKPOINT_MODEL_ID}"
    "--default_root_dir" "${ROOT_DIR}"
    "--inference_input_text" "'Danielle Harris had a main role in Super Capers, a 98 minute long movie.'"
)

cd src

# Check if $HARDWARE_SETTING starts with "s" for SLURM
if [[ $HARDWARE_SETTING == s* ]]; then
    # Run the inference with sbatch
    if [[ ${NUM_NODES} -gt 1 ]]; then
        sbatch --job-name=generate_grapher_${MODEL_VARIANT}_${HARDWARE_SETTING}_%A \
        --partition=${PARTITION} \
        --qos=${QOS} \
        --nodes=${NUM_NODES} \
        --gres=gpu:${NUM_GPUS_PER_NODE} \
        --ntasks-per-node=${NUM_GPUS_PER_NODE} \
        --time=${TIME} \
        --output=${EXEC_DIR}/slurm-%A_%a.out \
        --mail-type=BEGIN \
        --mail-user=${USER_MAIL} \
        main.slm "${python_args[@]}"
    else
        sbatch --job-name=generate_grapher_${MODEL_VARIANT}_${HARDWARE_SETTING}_%A \
        --partition=${PARTITION} \
        --qos=${QOS} \
        --gres=gpu:${NUM_GPUS_PER_NODE} \
        --ntasks-per-node=${NUM_GPUS_PER_NODE} \
        --time=${TIME} \
        --output=${EXEC_DIR}/slurm-%A_%a.out \
        --mail-type=BEGIN \
        --mail-user=${USER_MAIL} \
        main.slm "${python_args[@]}"
    fi
else
    # Run the inference without sbatch
    python3 main.py "${python_args[@]}"
fi