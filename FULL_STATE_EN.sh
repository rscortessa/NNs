#!/bin/bash

# Read variables from console
L=$1
NN=$2
NR=$3
NSPCA=$4
Nangle=$5
NMEAN=$6
G_0=$7
G_1=$8
G_2=$9

# Loop over ii_angle from 0 to Nangle

for (( ii_angle=0; ii_angle<=Nangle; ii_angle++ ))
do
    echo "Running angle $ii_angle"
    python ANGLE_BY_ANGLE_OPTIMIZED_FULL_STATE_QIM_RBM.py $L $NN $NR $NSPCA $Nangle $NMEAN $ii_angle $G_0 $G_1 $G_2 &
    
done

