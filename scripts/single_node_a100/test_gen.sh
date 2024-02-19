#!/bin/bash

# Load environment variables
cd ../../src && source .env 

python3 main.py --run test \
                --edges_as_classes 0 \
                --dataset webnlg \
                --data_path ${STORAGE_DRIVE}/data/core/webnlg-dataset/release_v3.0/en \
                --num_data_workers 15 \
                --checkpoint_model_id -1 \
                --batch_size 20 \
                --default_root_dir ${STORAGE_DRIVE}/data/core/grapher/output \
                --accelerator gpu \
                --max_epochs 100 \
                --num_nodes 1 \
                --num_sanity_val_steps 0 \
                --fast_dev_run 0 \
                --overfit_batches 0 \
                --limit_train_batches 1.0 \
                --limit_val_batches 1.0 \
                --limit_test_batches 1.0 \
                --accumulate_grad_batches 10 \
                --log_every_n_steps 100 \
                --check_val_every_n_epoch 1 

# Problems with interpreting 0 as False in main.py
# --detect_anomaly 0 \