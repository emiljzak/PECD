#!/bin/bash

export OMP_NUM_THREADS=$nproc
export OMP_STACKSIZE=8000m
export KMP_STACKSIZE=8000m
export XDG_RUNTIME_DIR=$pwd
export DISPLAY=:0.0
ulimit -s unlimited

echo "Number of OMP threads :" $OMP_NUM_THREADS
echo "OMP stack size :" $OMP_STACKSIZE
echo "KMP stack size :" $KMP_STACKSIZE
echo "ulimit :" `ulimit`
echo "Current directory :" $pwd
echo "Executable :" $exec
echo "Job name :" $jobname
echo "Running on master node :" `hostname`
echo "Job ID :" $SLURM_JOB_ID
echo "Start time :" `date`

$exec 1 8 4 > log 2> err

echo "Finish time :" date
echo "DONE"
