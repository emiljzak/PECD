#!/bin/bash

export exec="python3 /beegfs/desy/group/cfel/cmi/zakemil/PECD/pecd/$3"
export jobname="pecd_$3"
export pwd=`pwd`

export jobtype="cfel-cmi,all"
export nproc=32 #`nproc --all`
export wclim=100
export nnodes=1

echo "Job type :" $jobtype
echo "Job name :" $jobname
echo "Requested time :" $wclim
echo "Requested number of nodes :" $nnodes
echo "Requested number of cores :" $nproc
echo "sbatch submit..."

sbatch --partition=$jobtype --requeue --ntasks=$nproc --time=$wclim:00:00 --job-name=$jobname --output=$2$jobname.o --error=$2$jobname.e \
       $pwd/run_python.sh $1 $2 $3

echo "Number of restarts:":$SLURM_RESTART_COUNT
#$pwd/run_python.sh
