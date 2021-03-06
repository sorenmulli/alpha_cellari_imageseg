#!/bin/sh
#BSUB -q gpuv100
##BSUB -q gputitanxpascal
#BSUB -gpu "num=1"
#BSUB -J myWork
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=10GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -u s183911@student.dtu.dk
#BSUB -R "select[gpu32gb]"
#BSUB -N

echo "Running script..."
python3 train.py