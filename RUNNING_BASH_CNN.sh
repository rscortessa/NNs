#!/bin/bash

# Generates 0, 1, 2... up to 9
a=$1
for i in {0..9}
do
   nohup python PLAYING_WITH_CNN_RUNNING.py > STUDY_i${a}N${i}.txt & 
   echo "Iteration: $i"
done
