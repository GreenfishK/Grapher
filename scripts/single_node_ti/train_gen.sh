#!/bin/bash

# Load environment variables
cd ../../src && source .env 

python3 main.py --run train \
                --edges_as_classes 0 \
                --dataset webnlg \
                --pretrained_model t5-large \
                --data_path ${STORAGE_DRIVE}/data/core/webnlg-dataset/release_v3.0/en \
                --cache_dir ${STORAGE_DRIVE}/data/core/cache/grapher \
                --num_data_workers 1 \
                --every_n_epochs 1 \
                --checkpoint_model_id -1 \
                --max_nodes 8 \
                --max_edges 7 \
                --default_seq_len_node 20 \
                --default_seq_len_edge 20 \
                --batch_size 8 \
                --lr 1e-2 \
                --focal_loss_gamma 3 \
                --dropout_rate 0.5 \
                --num_layers 2 \
                --eval_dump_only 0 \
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
