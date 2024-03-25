#!/usr/bin/env bash
#SBATCH --job-name=test_gen_%A 
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-23
#SBATCH --output=/gpfs/data/fs72332/fkovacev/data/core/grapher_repro/output/webnlg_version_2/test/slurm-%A_%a.out 
#SBATCH --mail-type=BEGIN 
#SBATCH --mail-user=filip.kovacevic@tuwien.ac.at 

cd ../
. .env

# run the test on experiment "webnlg_version_1" using latest checkpoint last.ckpt
srun python main.py --run test \
                    --version 2 \
                    --default_root_dir /gpfs/data/fs72332/fkovacev/data/core/grapher_repro/output \
                    --data_path /gpfs/data/fs72332/fkovacev/data/core/grapher_repro/webnlg-dataset/release_v3.0/en \
                    --accelerator "gpu" \
                    --devices 1