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

for ((i=0;i<=NMEAN;i++))
do
    for((j=start;j<=NANGLE;j++))
    do
	python V2_OPTUNA_OPTIMIZED_RUN_ENERGY.py  $L $NN $NR $NSPCA $NANGLE $NMEAN $j $G &      
    done
done
