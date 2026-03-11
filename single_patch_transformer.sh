#!/bin/bash
# Initialize variables (replace these placeholder integers with your actual values)
L=10
g=150
angles=(0 2 4 6 8 10 12) # List of integers
Nrep=10                   # Number of repetitions
architecture="spin_dependent_T"    #architecture (in this case is the transformer)
n_heads=1
head_dim=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
n_patches=1
NR=6000

for angle in "${angles[@]}"; do
    for h_dim in "${head_dim[@]}"; do
	python PLAYING_WITH_NN_OPTUNA_STUDY.py --L "$L" --g "$g" --angle "$angle" --architecture "$architecture" --n_heads "$n_heads" --head_dim "$h_dim" --n_patches "$n_patches" --clipping "True" &
    done
done
wait
echo "All trials are finished"

# Loop over each angle in the angles array


for ((niter=1; niter<=Nrep; niter++)); do
    
    # Loop Nrep times
    for angle in "${angles[@]}"; do
        for h_dim in "${head_dim[@]}"; do
            # Execute the python script in the background (& for parallel execution)
            # Note: Added the .py extension assuming it's a standard python file, remove if unnecessary.
            python PLAYING_WITH_NN_RUNNING.py --L "$L" --g "$g" --angle "$angle" --architecture "$architecture" --n_heads "$n_heads" --head_dim "$h_dim" --n_patches "$n_patches" --clipping "True" --NR "$NR" &
	    sleep 5
            echo "Launched $angle, $niter"
	done
    done
done

# Wait for all background processes to finish before exiting the script
wait

echo "All parallel jobs have been launched and completed."

