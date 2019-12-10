#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J myWork
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -u s183911@student.dtu.dk
#BSUB -R "select[gpu32gb]"
#BSUB -N
module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8

echo "Running script..."
python3 train.py