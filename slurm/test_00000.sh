#!/usr/bin/env bash

#SBATCH --job-name=pgto_test
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-short
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:rtxa6000:1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"

export PYTHONUNBUFFERED=1

echo "$(date +"%T")"
uv run -m offline.run --num-workers 1 --worker-id 0 --max-segment 0 --device cuda --verbose
echo "$(date +"%T")"
