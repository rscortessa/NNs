#!/bin/bash

L=$1
W=$2
N_samples=$3
Gamma=$4
Ntheta=$5
N_REP=$6

echo "Running with Gamma=$Gamma, L=$L, N_samples=$N_samples"

julia R_QIM.jl "$L" "$W" "$Gamma" "$Ntheta" "$N_samples" "$N_REP" 

dir=${MODEL}L${L}W${W}NS${N_samples}GI${Gamma}GF${Ntheta}NR${N_REP}
mkdir "${dir}"

mv *M5L${L}W${W}NS${N_samples}*.txt* "${dir}/"

