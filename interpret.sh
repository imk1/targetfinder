#!/bin/bash
#PBS -q long
#PBS -o /home/brianhie/bqtl/scripts_out/interpret.out
#PBS -e /home/brianhie/bqtl/scripts_out/interpret.err
#PBS -l nodes=1:ppn=5
#PBS -l mem=2gb

cd /home/brianhie/bqtl/targetfinder/

python interpret.py $TF/output-ep/training.h5 $METRIC > $TF/interpret/$METRIC.txt
