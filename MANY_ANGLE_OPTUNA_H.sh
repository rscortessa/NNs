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

for ((j=0;j<=NANGLE;j++))
do
    python V2_INIT_OPTUNA_STUDY_ENERGY.py $L $NN $NR $NSPCA $NANGLE $NMEAN $G $j
    for((i=start;i<=end;i++))
    do
	python V2_OPTUNA_STUDY_ENERGY.py  $L $NN $NR $NSPCA $NANGLE $NMEAN $ANGLE $j &
    done    
done
