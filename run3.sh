#!/bin/bash

G=$1

python mean_bid_conv.py 48 1 "$G" 1024 50 20 10 4 &
python mean_bid_conv.py 64 1 "$G" 1024 50 20 10 4 &
python mean_bid_conv.py 96 1 "$G" 1024 50 20 10 4 &
