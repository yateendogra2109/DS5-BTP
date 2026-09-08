#!/bin/bash
#SBATCH --job-name=pipeline_test
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=pipeline_test.out
#SBATCH --error=pipeline_test.err

source ~/.bashrc
conda activate modelmerge

echo "========================================"
echo "       Training Pipeline Temp Test"
echo "========================================"

echo "Starting Dry Run..."
torchrun --standalone --nnodes=1 --nproc_per_node=2 train_tf_vae.py \
    --dataset qwen_models \
    --batch_size 4 \
    --stage1_epochs 10 \
    --n_epochs 20 \
    --kl_weight 0.0001 \
    --wandb_mode disabled