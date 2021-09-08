import numpy as np
import CONSTANTS
import os
import itertools
def gen_input(jobtype):
    """ Set up essential input parameters"""

    params = {}

    """ ===== mode of execution ===== """ 
    """
        1) 'propagate':    propagate wavefunction for a grid of Euler angles and a grid of parameters
        2) 'analyze':      analyze wavefunction for a grid of Euler angles and a grid of parameters
    """
    
    params['mode']      = 'propagate'
    
    
    """ ===== type of job ===== """ 
    """
        1) 'local':    local job 
        2) 'slurm':    submission to a SLURM workload manager for an HPC job
    """

    params['jobtype'] 	= "local" #maxwell


    """ ===== molecule definition ====== """ 
    """
        Set up properties of the molecule, including atomic masses, geometry, MF embedding
        *)  mol_geometry: for now only supports a dictionary with internal coordinates (bond lengths, angles). 
            Extend later to the option of the cartesian input. 
        *) mol_embedding (string): define the MF embedding, which matches the embedding used for calculating the ro-vibrational wavefunctions.
            Extend later beyond the TROVE wavefunctions. Add own ro-vibrational wavefunctions and embeddings.
    """

    params['molec_name']    = "c"
    params['mol_geometry']  = {"rc":0.0} #angstroms
    params['mol_masses']    = {"c":12.0}
    params['mol_embedding'] = "bisector" #TROVE's bisector embedding



    """====== basis set parameters for BOUND ======"""
    """ 
        Set up basis set parameters for the construction of the stationary Hamiltonian matrix. 
        Bound states are calculated with these parameters.
        Format (tuple): params['bound_nnn'] = (par_min, par_max, number_of_params) - to set up loop over parameters
    """
    params['bound_nlobs']   = (10,10,1)
    params['bound_lmax']    = (2,2,1)
    params['bound_binw']    = (1.4,1.4,1)

    params['bound_nbins']   = 20
    params['bound_rshift']  = 0.0

    params['N_euler'] 	    = 1 #number of euler grid points per dimension for orientation averaging
    params['N_batches'] 	= 1 #number of batches for orientation averaging





    params['save_ham0']     = True #save the calculated bound state Hamiltonian
    params['save_psi0']     = True #save psi0
    params['save_enr0']     = True #save eigenenergies for psi0

    params['num_ini_vec']   = 20 # number of initial wavefunctions (orbitals) stored in file




    return params
