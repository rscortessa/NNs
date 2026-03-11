#!/bin/bash

# Initialize variables (replace these placeholder integers with your actual values)
L=10
g=50
angles=(-24 -22 -20 -18 -16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16 18 20 22 24) # List of integers
Nrep=10                   # Number of repetitions
architecture="RBM_COMPLEX"    #architecture (in this case is the transformer)
NR=2000
for angle in "${angles[@]}"; do
    python PLAYING_WITH_NN_OPTUNA_STUDY.py --L "$L" --g "$g" --angle "$angle" --architecture "$architecture"  &
done
wait
echo "All trials are finished"

# Loop over each angle in the angles array


for ((niter=1; niter<=Nrep; niter++)); do
    
    # Loop Nrep times
    for angle in "${angles[@]}"; do
        
        # Execute the python script in the background (& for parallel execution)
        # Note: Added the .py extension assuming it's a standard python file, remove if unnecessary.
        python PLAYING_WITH_NN_RUNNING.py --L "$L" --g "$g" --angle "$angle" --architecture "$architecture"   &
	sleep 5
        echo "Launched $angle, $niter"
    done
    
done

# Wait for all background processes to finish before exiting the script
wait

echo "All parallel jobs have been launched and completed."

