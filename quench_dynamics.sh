#!/bin/bash

# Initialize variables (replace these placeholder integers with your actual values)
L=10
g=50
angle=0 # List of integers
Nrep=10                   # Number of repetitions
architecture="RBM_COMPLEX"    #architecture (in this case is the transformer)
NN=2.0
t=(1 5 10 20 50 100 200 500 1000)
nt=100000000
NR=4000

for dt in "${t[@]}"; do
    python PLAYING_WITH_NN_OPTUNA_STUDY.py --L "$L" --g "$g" --angle "$angle" --architecture "$architecture" --NN "$NN" --model "QUENCH_QIM" --dt "$dt" --nt "$nt" &
done
				
wait
echo "All trials are finished"

# Loop over each angle in the angles array
for ((niter=1; niter<=Nrep; niter++)); do
    for dt in "${t[@]}"; do
    python PLAYING_WITH_NN_INF_RUNNING.py --L "$L" --g "$g" --angle "$angle" --architecture "$architecture" --NN "$NN" --model "QUENCH_QIM" --dt "$dt" --nt "$nt" &
    done
done

# Wait for all background processes to finish before exiting the script
wait

echo "All parallel jobs have been launched and completed."

