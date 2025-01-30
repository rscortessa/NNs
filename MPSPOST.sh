#!/bin/bash

# Prompt for L and N_samples
#read -p "Enter the value of L: " L
#read -p "Enter the number of samples (N_samples): " N_samples

# Prompt for Gamma range and increment
#read -p "Enter the starting value of Gamma: " Gamma_start
#read -p "Enter the ending value of Gamma: " Gamma_end
#read -p "Enter the increment for Gamma: " Gamma_increment
L=$1
W=$2
N_samples=$3
Gamma_start=$4
Gamma_end=$5
Gamma_increment=$6

echo "Value of L=$L and W=$W" 
echo "Starting value of Gamma=$Gamma_start"
echo "Ending value of Gamma=$Gamma_end"
echo "Increment of Gamma=$Gamma_increment"

# Loop over Gamma values
Gamma=$Gamma_start
while (( $(echo "$Gamma <= $Gamma_end" | bc -l) )); do
    echo "Running with Gamma=$Gamma, L=$L, N_samples=$N_samples"
#    julia DMRG_XYZ.jl "$L" "$W" "$Gamma" "$N_samples"
    python run_MPS.py "$L" "$W" "$Gamma" "$N_samples"
    python datasetMPS_run.py "$L" "$W" "$Gamma" "$N_samples"
    python dataset_density_run.py "$L" "$W" "$Gamma" "$N_samples" "1000" "5"
    Gamma=$(echo "$Gamma + $Gamma_increment" | bc -l)  # Increment Gamma
done
