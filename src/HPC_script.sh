#!/bin/sh
#BSUB -q gpuk80
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
 
echo "Running script..."
python3 test_print_slet_gerne.py
python3 model.py
echo "Submitted training model"
