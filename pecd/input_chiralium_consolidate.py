#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2022 Emil Zak <emil.zak@cfel.de>, <emil.j.zak@gmail.com>
#

import numpy as np
import graphics

def read_input():

    params = {}
 
    params['job_label']    = "test_bav" #job identifier. In case of Psi4 ESP it can be metod/basis specification: "UHF-aug-cc-pVTZ" #"UHF_6-31Gss"
    params['molec_name']   = "chiralium"
    params['bound_nlobs_arr']   = (10,10,1)
    params['bound_lmax_arr']    = (2,2,1)
    params['bound_binw_arr']    = (2.0,2.0,1)
    params['bound_nbins']       = 17
    params['prop_nbins']        = 50

    params['tmax']              = 1000.0 

    params['helicity_consolidate']          = ["L","R"]

    #maximum number of photons considered in PECD calculations based on Legendre expansion.
    params['Nmax_photons']      = 4

    params['N_euler'] 	        = 2   
    params['N_batches'] 	    = 2    
    params['orient_grid_type']  = "2D"  

    params['consolidate'] = {'bcoeffs': True}

    params['index_energy'] = [0]
    params['index_time'] = [1]
    params['index_bcoeff'] = list(np.linspace(0,4,5,endpoint=True,dtype=int))

    return params

