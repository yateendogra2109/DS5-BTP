#!/bin/bash
#SBATCH --job-name=pipeline_test
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=00:10:00
#SBATCH --output=pipeline_test.out
#SBATCH --error=pipeline_test.err

source ~/.bashrc
conda activate modelmerge

echo "========================================"
echo "       Training Pipeline Temp Test"
echo "========================================"

python test_pipeline.py