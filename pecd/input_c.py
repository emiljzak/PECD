import numpy as np
import CONSTANTS
import os
import itertools
def gen_input(jobtype):
    """ Set up essential input parameters"""

    params = {}

    """ ==== mode of execution ==== """ 
    """
        1) 'propagate':    propagate wavefunction for a grid of Euler angles and a grid of parameters
        2) 'analyze':      analyze wavefunction for a grid of Euler angles and a grid of parameters
    """
    
    params['mode']      = 'propagate'
    
    
    """ ==== type of job ==== """ 
    """
        1) 'local':    local job 
        2) 'slurm':    submission to a SLURM workload manager for an HPC job
    """

    params['jobtype'] 	= "local" #maxwell


    params['molec_name']        = "c"

    return params
