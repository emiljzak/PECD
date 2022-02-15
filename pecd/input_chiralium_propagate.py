#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2022 Emil Zak <emil.zak@cfel.de>, <emil.j.zak@gmail.com>
#

import numpy as np

def read_input():

    params = {}

    params['job_label']    = "test_orient" 
    
    """====== Basis set parameters ======"""
    """ 
        Set up basis set parameters for the calculation of the stationary Hamiltonian matrix. 
        Bound states are calculated with these parameters.
        Format (tuple): params['bound_nnn'] = (par_min, par_max, number_of_params) - to set up a loop over parameters
    """
    """ BOUND PART"""
    params['bound_nlobs_arr']   = (10,10,1)
    params['bound_lmax_arr']    = (2,2,1)
    params['bound_binw_arr']    = (2.0,2.0,1)

    params['bound_nbins']       = 10
    params['bound_rshift']      = 0.0

    """ PROPAGATION PART"""
    params['prop_nbins']        = 20


    """==== time-grid parameters ===="""
    params['time_units']    = "as"
    params['t0']            = 0.0 
    params['tmax']          = 40.0 
    params['dt']            = 4.0 # replace with the calculated number for N points per cycle
    params['wfn_saverate']  = 10 #save rate wrt. index labeling the timegrid. '1' means save all time-steps


    """==== Molecule-field orientation ===="""
    params['N_euler'] 	        = 2    # number of euler grid points per dimension (beta angle) for orientation averaging. Alpha and gamma are on double-sized grid.
    params['N_batches'] 	    = 1    # number of batches for orientation averaging
    params['orient_grid_type']  = "3D"  # 2D or 3D. Use 2D when averaging is performed over phi in W2D.

    """ ===== Molecule definition ====== """ 
    """
        Set up properties of the molecule, including atomic masses, geometry, MF embedding
        *)  mol_geometry: for now only supports a dictionary with internal coordinates (bond lengths, angles). 
            Extend later to the option of the cartesian input. 
        *) mol_embedding (string): define the MF embedding, which matches the embedding used for calculating the ro-vibrational wavefunctions.
            Extend later beyond the TROVE wavefunctions. Add own ro-vibrational wavefunctions and embeddings.
    """
    params['molec_name']    = "chiralium"
    params['mol_geometry']  = {"rc":0.0} #angstroms
    params['mol_masses']    = {"c":12.0}
    params['mol_embedding'] = "bisector" #TROVE's bisector embedding


    """ ====== Initial wavefunction ====="""
    params['ivec']              = 2 

    """===== kinetic energy matrix ====="""
    params['keo_method']        = "blocks"

    """===== potential energy matrix ====="""
    params['matelem_method']    = "analytic" 
    params['sph_quad_global']   = "lebedev_119" 
    params['sph_quad_tol']      = 1e-10     

    """==== electrostatic potential ===="""
    params['esp_mode']           = "anton" 
    params['multi_lmax']         = 8 
    params['esp_rotation_mode']  = 'mol_xyz' #'on_the_fly', 'to_wf'


    """===== Hamiltonian parameters ====="""
    params['hmat_filter']           = 1e-4 #threshold value (in a.u.) for keeping matrix elements of the field-free Hamiltonian
    params['num_ini_vec']           = 20 # number of initial wavefunctions (orbitals) stored in file
    params['map_type']              = 'DVR_NAT' #DVR, DVR_NAT, SPECT (mapping of basis set indices)

    """ ===== ARPACK eigensolver parameters ===== """
    params['ARPACK_tol']        = 1e-8      # error tolerance (relative)
    params['ARPACK_maxiter']    = 60000     # maximum number of iterations
    params['ARPACK_enr_guess']  = None      # energy guess for the shift inverse mode in (eV)
    params['ARPACK_which']      = 'SA'      # LA, SM, SA, LM
    params['ARPACK_mode']       = "normal"  # normal or inverse

    """ ====== FIELD PARAMETERS ====== """
    params['freq_units']    = "ev"      # nm or ev
    params['omega']         = 22.1   # 23.128 nm = 54 eV, 60 nm = 20 eV
    params['intensity']     = 5.0e+13   # W/cm^2: peak intensity
    params['field_form']    = "analytic" #or numerical (i.e. read from file). To be implemented.
    params['field_func_name']    = "LCPL"
    params['field_env_name']     = "gaussian" 

    """ gaussian pulse """
    params['gauss_tau']     = 1000.0/np.sqrt(2.0) #as: pulse duration (sigma). When e^-t^2/T^2 is used we divide by sqrt(2)
    params['gauss_t0']      = 3000.0 #as: pulse centre

    """ sin2 pulse """
    params['sin2_ncycles']  = 10
    params['sin2_t0']       = 2000.0

    params['CEP0']          = 0.0 #CEP phase of the field


    """ === ro-vibrational part ==== """ 
    params['density_averaging'] = False #use rotational proability density for orientation averageing. Otherwise uniform probability. 

    params['Jmax']              = 60 #maximum J for the ro-vibrational wavefunction
    params['rv_wavepacket_time']= 50
    params['rv_wavepacket_dt']  = 0.1 #richmol time-step in ps #

    """====  SAVING and PLOTTING ===="""
    params['save_psi0']     = True #save psi0
    params['save_enr0']     = True #save eigenenergies for psi0

    params['wavepacket_format'] = "h5" #dat or h5
    params['plot_elfield']      = False



    return params
