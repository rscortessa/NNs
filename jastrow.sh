#!/bin/bash

start=0
L=$1
NN=$2
NR=$3
NSPCA=$4
NMEAN=$5
end=$6
seed=$7


python OPTIMIZATION_SEARCH_JASTROW.py $L $NN $NR $NSPCA $NMEAN $end $seed

for ((i=start; i<end; i++))
do
  python OPTIMIZED_JASTROW_INFIDELITY_QIM_RBM.py $L $NN $NR $NSPCA $NMEAN $i $seed &
done
