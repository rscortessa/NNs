#!/bin/bash

L=$1
W=$2
Gamma=$3
Ntheta=$4
N_samples=$5
N_REP=$6
MODEL=R_XYZ
echo "Running with Gamma=$Gamma, L=$L, N_samples=$N_samples"

julia R_XYZ.jl "$L" "$W" "$Gamma" "$Ntheta" "$N_samples" "$N_REP" 

dir=${MODEL}L${L}W${W}NS${N_samples}GI${Gamma}NT${Ntheta}NR${N_REP}
mkdir "${dir}"

mv *M5L${L}W${W}*G${Gamma}*.txt* "${dir}/"

