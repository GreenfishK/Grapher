#!/bin/bash

# Load environment variables including HARDWARE_SETTING for the run
cd ../src 
source .env 

# Hyperparameters: Change them if needed. The env variables in the positional arguments are set in .env and set_hparams.sh
python_args=(
    "--run" "test"
    "--edges_as_classes" "${MODEL_VARIANT}"
    "--dataset" "webnlg"
    "--data_path" "${STORAGE_DRIVE}/data/core/webnlg-dataset/release_v3.0/en"
    "--num_data_workers" "${NUM_DATAWORKERS}"
    "--checkpoint_model_id" "-1"
    "--batch_size" "${BATCH_SIZE}"
    "--default_root_dir" "${STORAGE_DRIVE}/data/core/grapher/output"
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

# Check if $HARDWARE_SETTING starts with "s" for SLURM
if [[ $HARDWARE_SETTING == s* ]]; then
    if [[ ${NUM_NODES} -gt 1 ]]; then
        sbatch --job-name=train_grapher_${MODEL_VARIANT}_${HARDWARE_SETTING} \
        --partition=${PARTITION}\
        --qos=${QOS} \
        --nodes=${NUM_NODES} \
        --gres=gpu:${NUM_GPUS_PER_NODE} \
        --ntasks-per-node=${NUM_GPUS_PER_NODE}\
        --time=1-0 \
        --mail-type=BEGIN \
        --mail-user=filip.kovacevic@tuwien.ac.at \
        main.slm "${python_args[@]}"
    else
        sbatch --job-name=train_grapher_${MODEL_VARIANT}_${HARDWARE_SETTING} \
        --partition=${PARTITION}\
        --qos=${QOS} \
        --gres=gpu:${NUM_GPUS_PER_NODE} \
        --ntasks-per-node=${NUM_GPUS_PER_NODE}\
        --time=1-0 \
        --mail-type=BEGIN \
        --mail-user=filip.kovacevic@tuwien.ac.at \
        main.slm "${python_args[@]}"
    fi
else
    # Run the test without sbatch
    python3 main.py "${python_args[@]}"
fi