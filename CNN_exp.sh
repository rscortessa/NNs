
#!/bin/bash

# Initialize variables (replace these placeholder integers with your actual values)
L=10
g=150
angles=(12) # List of integers
Nrep=10                   # Number of repetitions
architecture="CNN_COMPLEX"    #architecture (in this case is the transformer)
kernel_size=10
features={2,10}
head_dim=$L
n_patches=5
NR=4000

for angle in "${angles[@]}"; do
    python PLAYING_WITH_NN_OPTUNA_STUDY.py --L "$L" --g "$g" --angle "$angle" --architecture "$architecture" --features 10 10 --kernel_size "$kernel_size" --padding "SAME"   &
done
wait
echo "All trials are finished"

# Loop over each angle in the angles array
for angle in "${angles[@]}"; do
    
    # Loop Nrep times
    for ((niter=1; niter<=Nrep; niter++)); do
        
        # Execute the python script in the background (& for parallel execution)
        # Note: Added the .py extension assuming it's a standard python file, remove if unnecessary.
        python PLAYING_WITH_NN_RUNNING.py --L "$L" --g "$g" --angle "$angle" --architecture "$architecture" --features 10 10 --kernel_size "$kernel_size" --padding "SAME" --NR "$NR"   &
        
    done
done

# Wait for all background processes to finish before exiting the script
wait

echo "All parallel jobs have been launched and completed."

