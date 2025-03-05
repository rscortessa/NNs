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

dir=${MODEL}L${L}W${W}NS${N_samples}GI${Gamma}GF${GammaF}NR${N_REP}
mkdir "${dir}"
cp ${dir}/DATA*MPS*.txt* .

echo "Running with Gamma=$Gamma, L=$L, N_samples=$N_samples"


for (( jj=1;jj<=N_REP;jj++))
do
    Gamma=$4  
    while (( $(echo "$Gamma <= $GammaF" | bc -l) )); do
        python run_MPS.py "$L" "$W" "$Gamma" "$N_samples" "$jj" 
        python dataset_density_run.py "$L" "$W" "$Gamma" "$N_samples" "1000" "5" "$jj"
	Gamma=$(echo "$Gamma + $DG" | bc -l)  # Increment Gamma
    done
    
done

wait
if "$Delete" ; then
    rm DATAM5L${L}W${W}NS${N_samples}NR1000G*.txt*
    rm DATAM5L${L}W${W}NS${N_samples}MPSG*.txt*
else
    mv *M5L${L}W${W}NS${N_samples}*.txt* "${dir}/"
fi

