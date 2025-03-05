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

echo "Running with Gamma=$Gamma, L=$L, N_samples=$N_samples"


M=4  # Number of parallel processes
chunk_size=$(( (N_REP + M - 1) / M ))  # Ceiling division


for j in $(seq 0 $((M-1))); do
    start=$(( j * chunk_size + 1 ))
    end=$(( (j + 1) * chunk_size ))
    if (( end > N_REP )); then
        end=$N_REP
    fi

    echo "Process $j working on $start to $end"

    (
        for (( jj=$start; jj<=$end; jj++ )); do
            Gamma=$4
            while (( $(echo "$Gamma <= $GammaF" | bc -l) )); do
                python run_MPS.py "$L" "$W" "$Gamma" "$N_samples" "$jj"
                python dataset_density_run.py "$L" "$W" "$Gamma" "$N_samples" "1000" "5" "$jj"
                Gamma=$(echo "$Gamma + $DG" | bc -l)  # Increment Gamma
            done
        done
    ) &  # Background process
done

wait  # Wait for all parallel processes to finish
echo "All jobs finished!"

if "$Delete" ; then
    rm DATAM5L${L}W${W}NS${N_samples}NR1000G*.txt*
    rm DATAM5L${L}W${W}NS${N_samples}MPSG*.txt*
else
    mv *M5L${L}W${W}NS${N_samples}*.txt* "${dir}/"
fi

