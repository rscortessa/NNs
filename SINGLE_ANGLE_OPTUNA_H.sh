#!/bin/bash

start=0
end=4
L=$1
NN=$2
NR=$3
NSPCA=$4
NANGLE=$5
NMEAN=$6
G=$7
ANGLE=$8

python V2_INIT_OPTUNA_STUDY_ENERGY.py $L $NN $NR $NSPCA $NANGLE $NMEAN $G $ANGLE

for ((i=start; i<=end; i++))
do
    python V2_OPTUNA_STUDY_ENERGY.py  $L $NN $NR $NSPCA $NANGLE $NMEAN $ANGLE $G &  
done
