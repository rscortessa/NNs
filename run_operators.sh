#!/bin/bash
N=$1
Gamma=$2
GammaF=$3
n_sample=$4
n_run=$5
n_mean=$6
each=$7
NG=$8
NN=$9
ex=$10
# sbatch -n4
# sbatch -p regular2 
source /home/rcortess/anaconda3/etc/profile.d/conda.sh
conda activate workforce

python3 var_obv.py $N $Gamma $GammaF $n_sample $n_run $n_mean $each $NG $NN $ex

