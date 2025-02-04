#!/bin/bash

L=$1
W=$2
N_samples=$3
Gamma=$4
N_rep=$5

for jj in $(seq 1 $N_rep)
do
  bash ONLY_XYZ.sh $L $W $N_samples $Gamma
  mv "M5L${L}W${W}NS${N_samples}NR1000G${Gamma}D.txt" "M5L${L}W${W}NS${N_samples}NR1000G${Gamma}D.txt$jj"
done
