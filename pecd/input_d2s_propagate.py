#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2022 Emil Zak <emil.zak@cfel.de>, <emil.j.zak@gmail.com>
#

import numpy as np

def read_input():

    params = {}

    params['job_label']    = "no_com" 
    
    """====== Basis set parameters ======"""
    """ 
        Set up basis set parameters for the calculation of the stationary Hamiltonian matrix. 
        Bound states are calculated with these parameters.
        Format (tuple): params['bound_nnn'] = (par_min, par_max, number_of_params) - to set up a loop over parameters
    """
    """ BOUND PART"""
    params['bound_nlobs_arr']   = (12,12,1)
    params['bound_lmax_arr']    = (4,4,1)
    params['bound_binw_arr']    = (2.5,2.5,1)

    params['bound_nbins']       = 20
    params['bound_rshift']      = 0.0

    """ PROPAGATION PART"""
    params['prop_nbins']        = 20


    """==== time-grid parameters ===="""
    params['time_units']    = "as"
    params['t0']            = 0.0 
    params['tmax']          = 20.0 
    params['dt']            = 5.0 # replace with the calculated number for N points per cycle
    params['wfn_saverate']  = 1 #save rate wrt. index labeling the timegrid. '1' means save all time-steps


    """==== Molecule-field orientation ===="""
    params['restart']           = False #restart some of the runs?
    params['restart_mode']      = "file" #manual
    params['restart_helicity']  = "L"
    if params['restart_mode']      == "manual":
        params['restart_list']      =  []#global id list of jobs to be restarted. 
    params['N_euler'] 	        = 1   # number of euler grid points per dimension (beta angle) for orientation averaging. Alpha and gamma are on double-sized grid.
    params['N_batches'] 	    = 1    # number of batches for orientation averaging
    params['orient_grid_type']  = "1D"  # 2D or 3D. Use 2D when averaging is performed over phi in W2D.

    """ ===== Molecule definition ====== """ 

    params['molec_name']    = "d2s"
    params['mol_geometry']  = {"rSD1":1.336, "rSD2": 1.336, "alphaDSD": 92.06} #angstroms#in next generation load internal geometry from file
    params['mol_masses']    = {"S":32.0, "D":2.0}
    params['mol_embedding'] = "bisector" #TROVE's bisector embedding


    """ ====== Initial wavefunction ====="""
    params['ivec']              = 8 

    """===== kinetic energy matrix ====="""
    params['keo_method']        = "blocks"

    """===== potential energy matrix ====="""
    params['matelem_method']    = "lebedev" 
    params['sph_quad_global']   = "lebedev_041" 
    params['sph_quad_tol']      = 1e-10     
    params['r_cutoff']          = 40.0

    """==== electrostatic potential ===="""
    params['esp_mode']           = "psi4" 
    params['esp_rotation_mode']  = 'mol_xyz' #'on_the_fly', 'to_wf'
    params['scf_basis']          = 'aug-cc-pVTZ' #"cc-pDTZ" #"631G**"
    params['scf_method']         = 'UHF'
    params['scf_enr_conv']       = 1.0e-6 #convergence threshold for SCF
    params['multi_lmax']         = 8 

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
    params['omega']         = 5.0   # 23.128 nm = 54 eV, 60 nm = 20 eV
    params['intensity']     = 2.0e+13   # W/cm^2: peak intensity
    params['field_form']    = "analytic" #or numerical (i.e. read from file). To be implemented.
    params['field_func_name']    = "RCPL"
    params['field_env_name']     = "gaussian" 
    params['CEP0']               = 0.0 #CEP phase of the field

    """ gaussian pulse """
    params['gauss_tau']     = 500.0/np.sqrt(2.0) #as: pulse duration (sigma). When e^-t^2/T^2 is used we divide by sqrt(2)
    params['gauss_t0']      = 1000.0 #as: pulse centre

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
