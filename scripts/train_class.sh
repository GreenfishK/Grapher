#!/usr/bin/env bash
#SBATCH --job-name=train_class_%A 
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-23
#SBATCH --output=/gpfs/data/fs72332/fkovacev/data/core/grapher_repro/output/slurm-%A_%a.out
#SBATCH --mail-type=BEGIN 
#SBATCH --mail-user=filip.kovacevic@tuwien.ac.at 

cd ../
. .env
srun python main.py --version 1 \
                  --default_root_dir /gpfs/data/fs72332/fkovacev/data/core/grapher_repro/output \
                  --data_path /gpfs/data/fs72332/fkovacev/data/core/grapher_repro/webnlg-dataset/release_v3.0/en \
                  --cache_dir /gpfs/data/fs72332/fkovacev/data/core/cache/grapher \
                  --run train \
                  --max_epochs 100 \
                  --accelerator gpu \
                  --num_nodes 1 \
                  --num_data_workers 15 \
                  --lr 1e-4 \
                  --batch_size 10 \
                  --num_sanity_val_steps 0 \
                  --fast_dev_run 0 \
                  --overfit_batches 0 \
                  --limit_train_batches 1.0 \
                  --limit_val_batches 1.0 \
                  --limit_test_batches 1.0 \
                  --accumulate_grad_batches 10 \
                  --detect_anomaly True \
                  --log_every_n_steps 100 \
                  --val_check_interval 1000 \
                  --checkpoint_step_frequency 1000 \
                  --focal_loss_gamma 3 \
                  --dropout_rate 0.5 \
                  --num_layers 2 \
                  --edges_as_classes 1 \
                  --checkpoint_model_id -1 
