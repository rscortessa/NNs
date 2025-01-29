#!/bin/bash

G=$1

python single_dataset_conv.py 48 1 "$G" 1024 50 20 90 4 &
python single_dataset_conv.py 64 1 "$G" 1024 50 20 90 4 &
python single_dataset_conv.py 96 1 "$G" 1024 50 20 90 4 &
