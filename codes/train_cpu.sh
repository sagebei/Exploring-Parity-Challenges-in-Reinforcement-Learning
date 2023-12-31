#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 1        # Request cores (8 per GPU)
#$ -l h_vmem=8G   # Request RAM (7.5GB per core)
#$ -l h_rt=24:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -N value      # Name for the job (optional)
#$ -t 1-10

# Load the necessary modules
module load python/3.8.5

# Load the virtualenv containing the pytorch package
source ~/venv/bin/activate

python main.py --n_elems=$1 \
               --n_train_samples=100000 \
               --n_eval_samples=5000 \
               --n_epochs=10000 \
               --noisy_label=$2 \
               --n_layers=1 \
               --noise='.' \
               --seed=${SGE_TASK_ID}  \
               --lr=0.3