#!/bin/bash

start=0
L=$1
NN=$2
NR=$3
NSPCA=$4
NANGLE=$5
NMEAN=$6
G=$7
angle=$8


for ((i=start; i<=NMEAN; i++))
do
    sleep $(python -c "import random; print(random.uniform(1.0, 3.0))")
    
    python V2_OPTUNA_OPTIMIZED_RUN_ENERGY.py  $L $NN $NR $NSPCA $NANGLE $NMEAN $angle $G &

done
