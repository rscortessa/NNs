#!/bin/bash

L=$1
W=$2
N_samples=$3
Gamma=$4
Delete=false
echo "Running with Gamma=$Gamma, L=$L, N_samples=$N_samples"

julia DMRG_XYZ.jl "$L" "$W" "$Gamma" "$N_samples"
python run_MPS.py "$L" "$W" "$Gamma" "$N_samples"
python dataset_density_run.py "$L" "$W" "$Gamma" "$N_samples" "1000" "5"

if "$Delete" ; then
    rm "DATAM5L${L}W${W}NS${N_samples}NR1000G${Gamma}.txt"
    rm "DATAM5L${L}W${W}NS${N_samples}MPSG${Gamma}.txt"
fi
