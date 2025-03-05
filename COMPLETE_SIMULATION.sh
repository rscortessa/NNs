#!/bin/bash

L=$1
W=$2
N_samples=$3
Gamma=$4
GammaF=$5
DG=$6
N_REP=$7
MODEL=$8
Delete=$9

bash PARALLEL_MODEL.sh "$L" "$W" "$N_samples" "$Gamma" "$GammaF" "$DG" "$N_REP" "$MODEL" "$Delete"
echo "PROCCESSING PART...."
bash SAVER_MODEL.sh "$L" "$W" "$N_samples" "$Gamma" "$GammaF" "$DG" "$N_REP" "$MODEL" "$Delete"
