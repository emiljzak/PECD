import numpy as np
import os
import subprocess

def run_propagate(N_euler,N_batches):

	path = os.getcwd()
	print ("The current working directory is %s" % path)
	START_FILES = os.listdir(path+"/slurm_run")
	os.chdir(path+"/slurm_run")
	for ibatch in range(N_batches):

		subprocess.call("./master_script.sh " + str(ibatch) + " " + str(N_batches) + " " + str(N_euler) , shell=True) 



N_euler = 3 #number of euler grid points per dimension
N_batches = 3

run_propagate(N_euler,N_batches)