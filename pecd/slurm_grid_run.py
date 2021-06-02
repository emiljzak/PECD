import numpy as np
import os

def run_propagate(N_euler,N_batches):

	path = os.getcwd()
	print ("The current working directory is %s" % path)
	START_FILES = os.listdir(path+"/slurm_run")

	for ibatch in range(N_batches):

		subprocess.call(START_FILES + "./master_script.sh " + str(ipoint) + " " + N_batches + " " + N_euler , shell=True) 



N_euler = 6
N_batches = 2
