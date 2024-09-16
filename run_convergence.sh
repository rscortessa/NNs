#!/bin/bash
N=$1
Gamma=$2
V=$3
n_sample=$4
n_run=$5
n_mean=$6
each=$7
k=$8

# sbatch -n4
# sbatch -p regular2 
source /home/rcortess/anaconda3/etc/profile.d/conda.sh
conda activate workforce

python3 convergence_obv.py $N $Gamma $V $n_sample $n_run $n_mean $each $k


