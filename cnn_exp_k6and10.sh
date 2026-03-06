#!/bin/bash

# Initialize variables (replace these placeholder integers with your actual values)
L=10
g=150
angles=(12) # List of integers
Nrep=10                   # Number of repetitions
architecture="CNN_COMPLEX"    #architecture (in this case is the transformer)
kernel_size=(6 10)
feat_a=(2 4 6 10)
feat_b=10
NR=6000

for angle in "${angles[@]}"; do
    for feat_1 in "${feat_a[@]}"; do
		  for kernel in "${kernel_size[@]}"; do
		      python PLAYING_WITH_NN_OPTUNA_STUDY.py --L "$L" --g "$g" --angle "$angle" --architecture "$architecture" --features "$feat_1" "$feat_b" --kernel_size "$kernel" --padding "SAME" --NR "$NR"   &
		  done
    done
done
				
wait
echo "All trials are finished"

# Loop over each angle in the angles array
for ((niter=1; niter<=Nrep; niter++)); do
    for angle in "${angles[@]}"; do
	for feat_1 in "${feat_a[@]}"; do
	    for kernel in "${kernel_size[@]}"; do
		python PLAYING_WITH_NN_RUNNING.py --L "$L" --g "$g" --angle "$angle" --architecture "$architecture" --features "$feat_1" "$feat_b" --kernel_size "$kernel" --padding "SAME" --NR "$NR"  &
		sleep 5
		echo "Launched $angle, $niter"
	    done
	done
    done
done

# Wait for all background processes to finish before exiting the script
wait

echo "All parallel jobs have been launched and completed."

