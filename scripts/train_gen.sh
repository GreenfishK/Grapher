#!/usr/bin/env bash
cd ../
# Load environment variables
source .env

python3 main.py   --version 2 \
                  --run train \
                  --num_data_workers 4 \
                  --lr 1e-4 \
                  --batch_size 8 \
                  --data_path ${STORAGE_DRIVE}/data/core/webnlg-dataset/release_v3.0/en \
                  --cache_dir ${STORAGE_DRIVE}/data/core/cache/grapher \
                  --default_root_dir ${STORAGE_DRIVE}/data/core/grapher/output \
                  --checkpoint_step_frequency 1000 \
                  --focal_loss_gamma 3 \
                  --dropout_rate 0.5 \
                  --num_layers 2 \
                  --edges_as_classes 0 \
                  --checkpoint_model_id -1 \
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
                  --val_check_interval 1000 \

# Problems with interpreting 0 as False in main.py
# --detect_anomaly 0 \

# --checkpoint_step_frequency 1000 \
# --val_check_interval 1000 \
# --max_epochs 100 \