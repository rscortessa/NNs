#!/bin/bash
N=$1
Gamma=$2
n_sample=$3
n_run=$4
n_mean=$5
each=$6
k=$7

# sbatch -n4
# sbatch -p regular2 
source /home/rcortess/anaconda3/etc/profile.d/conda.sh
conda activate workforce

python3 convergence_obv.py $N $Gamma $n_sample $n_run $n_mean $each $k


