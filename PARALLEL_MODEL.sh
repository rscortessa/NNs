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

echo "Running with Gamma=$Gamma, L=$L, N_samples=$N_samples"

julia DMRG_MODELS.jl "$L" "$W" "$Gamma" "$GammaF" "$DG" "$N_samples" "$N_REP" "$MODEL"

