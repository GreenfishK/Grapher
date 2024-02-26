#!/bin/bash

# Load environment variables including HARDWARE_SETTING for the run
cd ..
source .env 

# Hyperparameters: Change them if needed. The env variables in the positional arguments are set in .env and set_hparams.sh
python_args=(
    "--run" "test"
    "--edges_as_classes" "${MODEL_VARIANT}"
    "--dataset" "webnlg"
    "--data_path" "${DATA_DIR}"
    "--num_data_workers" "${NUM_DATAWORKERS}"
    "--checkpoint_model_id" "${CHECKPOINT_MODEL_ID}"
    "--batch_size" "${BATCH_SIZE}"
    "--default_root_dir" "${ROOT_DIR}"
    "--accelerator" "gpu"
    "--max_epochs" "100"
    "--num_nodes" "${NUM_NODES}"
    "--num_sanity_val_steps" "0"
    "--fast_dev_run" "0"
    "--overfit_batches" "0"
    "--limit_train_batches" "1.0"
    "--limit_val_batches" "1.0"
    "--limit_test_batches" "1.0"
    "--accumulate_grad_batches" "10"
    "--log_every_n_steps" "100"
    "--check_val_every_n_epoch" "1"
)

cd src

# Check if $HARDWARE_SETTING starts with "s" for SLURM
if [[ $HARDWARE_SETTING == s* ]]; then
    if [[ ${NUM_NODES} -gt 1 ]]; then
        sbatch --job-name=train_grapher_${MODEL_VARIANT}_${HARDWARE_SETTING}_%A \
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
        sbatch --job-name=train_grapher_${MODEL_VARIANT}_${HARDWARE_SETTING}_%A \
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
    # Run the test without sbatch
    python3 main.py "${python_args[@]}"
fi