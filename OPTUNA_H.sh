#!/bin/bash

start=0
end=5
L=$1
NN=$2
NR=$3
NSPCA=$4
NANGLE=$5
NMEAN=$6
G=$7


python INIT_OPTUNA_STUDY_ENERGY.py $L $NN $NR $NSPCA $NANGLE $NMEAN $G 

for ((i=start; i<=NANGLE; i++))
do
    for ((j=start; j<=end;j++))
	do
	    python OPTUNA_STUDY_ENERGY.py  $L $NN $NR $NSPCA $NANGLE $NMEAN $i $G &
    done
    
done
