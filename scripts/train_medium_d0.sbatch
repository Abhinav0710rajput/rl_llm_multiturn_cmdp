#!/bin/bash
#SBATCH --job-name=medium-d0
#SBATCH --partition=c24m170-a100-2
#SBATCH --account=ds_ga_1012-2026sp
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/train_medium_d0_%j.out

source ~/.bashrc
conda activate rl-clarify

cd $HOME/rl_llm_multiturn_project

PYTHONUNBUFFERED=1 python scripts/train.py --d1 0 --verbose \
    training.n_iterations=10 \
    training.rollout_batch_size=64 \
    training.eval_interval=5 \
    training.save_interval=5
