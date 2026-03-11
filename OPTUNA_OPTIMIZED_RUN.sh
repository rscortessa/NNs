#!/bin/bash

start=0
L=$1
NN=$2
NR=$3
NSPCA=$4
NANGLE=$5
NMEAN=$6
G=$7



for ((i=start; i<=NANGLE; i++))
do
    for ((j=start; j<=NMEAN;j++))
	do
	    python OPTUNA_OPTIMIZED_RUN_ENERGY.py  $L $NN $NR $NSPCA $NANGLE $NMEAN $i $G &
	    
    done
    
done
