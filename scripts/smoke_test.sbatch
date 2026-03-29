#!/bin/bash
#SBATCH --job-name=smoke-test
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=ds_ga_1012-2026sp
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_%j.out

source ~/.bashrc
conda activate rl-clarify

cd $HOME/rl_llm_multiturn_project
python scripts/smoke_test.py --two-gpu
