#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2022 Emil Zak <emil.zak@cfel.de>, <emil.j.zak@gmail.com>
#

import numpy as np
import graphics

def read_input():

    params = {}
 
    params['job_label']    = "orientations" #job identifier. In case of Psi4 ESP it can be metod/basis specification: "UHF-aug-cc-pVTZ" #"UHF_6-31Gss"
    params['molec_name']   = "d2s"
    params['bound_nlobs_arr']   = (6,6,1)
    params['bound_lmax_arr']    = (2,2,1)
    params['bound_binw_arr']    = (2.0,2.0,1)
    params['bound_nbins']       = 20
    params['prop_nbins']        = 50

    params['tmax']              = 2000.0 

    params['helicity_consolidate']  = ["L","R"]
    params['check_sizes']           = False #pre-check sizes of wavepacket files?
    params['res_jobs_per_batch']    = 10 #number of restart jobs per batch
    
    #maximum number of photons considered in PECD calculations based on Legendre expansion.
    params['Nmax_photons']      = 2
    params['N_euler'] 	        = 6   
    params['N_batches'] 	    = 72    
    params['orient_grid_type']  = "3D"  

    params['index_energy']  = [0]
    params['index_time']    = [0]
    params['index_bcoeff']  = list(np.linspace(0,4,5,endpoint=True,dtype=int))

    """ === ro-vibrational part ==== """ 
    params['density_averaging']     = True #use rotational proability density for orientation averageing. Otherwise uniform probability. 
    params['Jmax']                  = 60 #maximum J for the ro-vibrational wavefunction
    params['rv_wavepacket_name']    = "wavepacket_J60.h5"
    params['rv_coefficients_name']  = "coefficients_j0_j60.rchm"
    params['rv_wavepacket_time']    = 50
    params['rv_wavepacket_dt']      = 0.1 #richmol time-step in ps #


    return params

