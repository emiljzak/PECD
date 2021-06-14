#!/bin/bash

module load anaconda/3
#export exec="python3 /home/yachmena/RICHMOL/richmol/richmol.py"
export exec="python3 /gpfs/cfel/cmi/scratch/user/zakemil/PECD/pecd/PROPAGATE.py"
export jobname="pecd_run"
export pwd=`pwd`

export jobtype="cfel-cmi,cfel,all"
export nproc=32 #`nproc --all`
export wclim=100
export nnodes=1

echo "Job type :" $jobtype
echo "Job name :" $jobname
echo "Requested time :" $wclim
echo "Requested number of nodes :" $nnodes
echo "Requested number of cores :" $nproc
echo "sbatch submit..."

sbatch --partition=$jobtype --ntasks=$nproc --time=$wclim:00:00 --job-name=$jobname --output=$jobname.o --error=$jobname.e \
       $pwd/run_python.sh $1 $2 $3 $4 $5

#$pwd/run_python.sh
