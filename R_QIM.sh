#!/bin/bash

L=$1
W=$2
Gamma=$3
Ntheta=$4
N_samples=$5
N_REP=$6
MODEL=R_QIM
echo "Running with Gamma=$Gamma, L=$L, N_samples=$N_samples"

julia R_QIM.jl "$L" "$W" "$Gamma" "$Ntheta" "$N_samples" "$N_REP" 


