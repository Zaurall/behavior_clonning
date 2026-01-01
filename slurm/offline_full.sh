#!/usr/bin/env bash

#SBATCH --job-name=offline_full
#SBATCH --output=logs/%A_%a.out
#SBATCH --partition=gpu-long
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00

echo "Starting Optuna worker ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "SLURM assigned GPU devices: ${CUDA_VISIBLE_DEVICES}"
echo "GPU device count: $(nvidia-smi --list-gpus | wc -l)"

uv run -m offline.run --num-workers 6 --worker-id $SLURM_ARRAY_TASK_ID --max-segment 4999 --device cuda
