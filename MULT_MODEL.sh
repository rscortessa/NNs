#!/bin/bash

L=$1
W=$2
N_samples=$3
Gamma=$4
N_rep=$5
MODEL=$6
dir=$7
Delete=false

for jj in $(seq 1 $N_rep)
do
  bash ONLY_MODEL.sh $L $W $N_samples $Gamma $MODEL $Delete
  mv "M5L${L}W${W}NS${N_samples}NR1000G${Gamma}D.txt" "${dir}/M5L${L}W${W}NS${N_samples}NR1000G${Gamma}D.txt$jj"
  if [! "$Delete"] ; then
      mv "DATAM5L${L}W${W}NS${N_samples}NR1000G${Gamma}.txt" "${dir}/DATAM5L${L}W${W}NS${N_samples}NR1000G${Gamma}.txt$jj"
      mv "DATAM5L${L}W${W}NS${N_samples}MPSG${Gamma}.txt" "${dir}/DATAM5L${L}W${W}NS${N_samples}MPSG${Gamma}.txt$jj"
  fi
done
