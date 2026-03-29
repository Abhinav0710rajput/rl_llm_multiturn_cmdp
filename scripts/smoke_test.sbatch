#!/bin/bash
#SBATCH --job-name=smoke-test
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_%j.out

module purge
module load anaconda3/2024.06
conda activate rl-clarify

cd $HOME/rl_llm_multiturn_project
python scripts/smoke_test.py
