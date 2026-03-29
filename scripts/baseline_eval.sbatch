#!/bin/bash
#SBATCH --job-name=baseline-eval
#SBATCH --partition=c24m170-a100-2
#SBATCH --account=ds_ga_1012-2026sp
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/baseline_%j.out

source ~/.bashrc
conda activate rl-clarify

cd $HOME/rl_llm_multiturn_project
python scripts/baseline_eval.py --n 30 --two-gpu --multi-turn
