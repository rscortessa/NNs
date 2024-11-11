#!/bin/bash

# Prompt for L and N_samples
read -p "Enter the value of L: " L
read -p "Enter the number of samples (N_samples): " N_samples

# Prompt for Gamma range and increment
read -p "Enter the starting value of Gamma: " Gamma_start
read -p "Enter the ending value of Gamma: " Gamma_end
read -p "Enter the increment for Gamma: " Gamma_increment

# Loop over Gamma values
Gamma=$Gamma_start
while (( $(echo "$Gamma <= $Gamma_end" | bc -l) )); do
    echo "Running with Gamma=$Gamma, L=$L, N_samples=$N_samples"
    python run_MPS.py "$L" "$Gamma" "$N_samples"
    Gamma=$(echo "$Gamma + $Gamma_increment" | bc -l)  # Increment Gamma
done
