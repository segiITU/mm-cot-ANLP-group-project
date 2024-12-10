#!/bin/bash
#SBATCH --job-name=mm-cot-combiner
#SBATCH --output=%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=brown
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2

export SLURM_NTASKS=1
export SLURM_NTASKS_PER_NODE=1

# Modules and environment
module purge
module load Anaconda3
source activate mm-cot

# Run the Python script
CUDA_VISIBLE_DEVICES=0,1 python combiner.py \
    --generation_max_length 256 \
    --batch_size 4 \
    --model_path "models/mm-cot-large-rationale/mm-cot-large-rationale/" \
    --output_dir "Saving"