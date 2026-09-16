#!/bin/bash
#SBATCH --job-name=oasis_vae
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=part4_1_%j.out
#SBATCH --error=part4_1_%j.err

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$(dirname "$0")"
python part4_1.py --epochs 30 --batch-size 64 --latent-dim 2
