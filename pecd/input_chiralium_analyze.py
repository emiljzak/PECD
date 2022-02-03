#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2022 Emil Zak <emil.zak@cfel.de>, <emil.j.zak@gmail.com>
#

import numpy as np
import graphics

def read_input():

    params = {}
 
    params['job_label']    = "proptest2" #job identifier. In case of Psi4 ESP it can be metod/basis specification: "UHF-aug-cc-pVTZ" #"UHF_6-31Gss"
    
    """====== Basis set parameters ======"""
   
    """ BOUND PART """
    params['bound_nlobs_arr']   = (10,10,1)
    params['bound_lmax_arr']    = (2,2,1)
    params['bound_binw_arr']    = (2.0,2.0,1)

    params['bound_nbins']       = 20

    """ PROPAGATION PART """
    params['prop_nbins']        = 50

    """==== time-grid parameters ===="""
    params['time_units']    = "as"
    params['t0']            = 0.0 
    params['tmax']          = 100.0 
    params['dt']            = 2.0 # replace with the calculated number for N points per cycle
    params['wfn_saverate']  = 1 #save rate wrt. index labeling the timegrid. '1' means save all time-steps


    """==== Molecule-field orientation ===="""

    params['N_euler'] 	        = 1    # number of euler grid points per dimension (beta angle) for orientation averaging. Alpha and gamma are on double-sized grid.
    params['N_batches'] 	    = 1    # number of batches for orientation averaging
    params['orient_grid_type']  = "3D"  # 2D or 3D. Use 2D when averaging is performed over phi in W2D.
    

    params['space_analyze_times']    =   list(np.linspace(0.0, params['tmax'], 4 ))
    params['momentum_analyze_times'] =   list(np.linspace(params['tmax'], params['tmax'], 1 ))

    params['analyze_space']     = [rho2D]
    params['analyze_momentum']  = [W2Dav]
    

    params['PECD']      = PECD
    params['W2Dav']     = W2Dav
    params['W2D']       = W2D
    params['rho2D']     = rho2D
    params['bcoeffs']   = bcoeffs


    load_observables()
    
    rho2D = {   'name':         'rho2D',
                'plane':        ('XY',), #in which Cartesian planes do we want to plot rho2D? 'XY','XZ','YZ' or [nx,ny,nz] - vector normal to the plane
                'plot':         (True, graphics.gparams_rho2D_polar()), #specify parameters of the plot to load
                'show':         True, # show image on screen                    
                'save':         True,
                'scale':        "log", #unit or log
                'r_grid':       {   'type':'manual', #manual or automatic grid type. 
                                    'npts': 800,    #ignored when automatic (2*rmax)
                                    'rmin': 0.0,    #ignored when automatic
                                    'rmax': 200.0  #ignored when automatic
                                    #Automatic means that we choose ranges based on maximum range given by the basis set.   
                                },                   
                'th_grid':      (0.0,2.0*np.pi,360),
                'coeff_thr':    1e-10 #threshold for the wavefunction coefficients in the calculation of rho
                }

    W2D = {     'name':         'W2D',
                'plane':        ('XZ',), #in which Cartesian planes do we want to plot rho2D? 'XY','XZ','YZ' or [nx,ny,nz] - vector normal to the plane
                'plot':         (True, graphics.gparams_W2D_polar()), #specify parameters of the plot to load
                'show':         False, # show image on screen
                'save':         True, # save array in file
                                # Momentum grid parameters only for plotting purposes
                'k_grid':       {   'type':'manual', #manual or automatic grid type. 
                                    'npts': 5000,    #ignored when automatic (2*rmax)
                                    'kmin': 0.0,    #ignored when automatic
                                    'kmax': 2.0  #ignored when automatic
                                    #Automatic means that we choose ranges based on maximum range given by the basis set.   
                                },                   
                'th_grid':      (0.0,2.0*np.pi,360),
                'legendre':     False, # calculate Legendre decomposition

                'PES':          True, # calculate PES
                'PES_params': {     
                                'name':         'PES',
                                'plot':         (True, graphics.gparams_PES()), #specify parameters of the plot to load
                                'show':         True, # show image on screen
                                'save':         True, # save array in file
                            
                                'k-axis':       "energy", # energy (eV) or momentum (a.u.)
                                'y-axis':       "log",   # log or unit scale
                                'normalize':    True,   # normalize the cross-section
                                },

                }


    W2Dav = {   'name':         'W2Dav',
                'plot':         (True, graphics.gparams_W2Dav_polar()), #specify parameters of the plot to load
                'show':         True, # show image on screen
                'save':         True, # save array in file
                                # Momentum grid parameters only for plotting purposes
                'k_grid':       {   'type':'automatic', #manual or automatic grid type. 
                                    'npts': 1000,    #ignored when automatic (2*rmax)
                                    'kmin': 0.0,    #ignored when automatic
                                    'kmax': 2.0  #ignored when automatic
                                    #Automatic means that we choose ranges based on maximum range given by the basis set.   
                                },                   
                'th_grid':      (0.0,2.0*np.pi,360),
                
                'nphi_pts':     1, #number of phi points for the integration over tha azimuthal angle.
                
                'legendre':     False, # calculate Legendre decomposition



                'PES':          True, # calculate PES
                'PES_params': {     
                                'name':         'PES',
                                'plot':         (True, graphics.gparams_PES()), #specify parameters of the plot to load
                                'show':         True, # show image on screen
                                'save':         True, # save array in file
                            
                                'k-axis':       "energy", # energy (eV) or momentum (a.u.)
                                'y-axis':       "log",   # log or unit scale
                                'normalize':    True,   # normalize the cross-section
                                },

                }

    PECD = {    'name':         'PECD',
                'plot2D':       (True, graphics.gparams_PECD2D()), #specify parameters of the plot to load
                'plot1D':       (True, graphics.gparams_PECD1D()), #specify parameters of the plot to load
                'orient_av':    False, # perform orientation averaging over the Euler angle's grid?
                'show':         True, # show images on screen
                'save':         True, # save arrays in files
                'kmax':         3.0,
                
                }

    bcoeffs = { 'name':     'bcoeffs',
                'plot':     (False, graphics.gparams_barray2D()),
                'show':     False,
                }

    #params['obs_params_rho'] = rho2D
    #params['obs_params_W2D'] = W2D
    #params['obs_params_W2Dav'] = W2Dav
    #params['obs_params_PECD'] = PECD



    """ *** Momentum-space wavefunction *** """
    params['FT_method']       = "FFT_hankel"    # "FFT_cart" #or quadratures
    # Fourier transform is calculated from the wavefunction calculated on real-space grid bounded by rcutoff and Rmax.
    params['npts_r_ft']       = 1000             # number of radial points over which the Hankel Transform is evaluated.
    params['rcutoff']         = 30.0            # radial cut-off of the wavepacket in the calculation of momentum space distributions
    
    params['plot_Plm']        = False           # plot and save photoelectron partial waves?
    params['plot_Flm']        = False           # plot and save individual Hankel transforms?

    """ *** Legendre expansion *** """
    params['Leg_lmax']          = 6      # maximum angular momentum in the Legendre expansion
    params['Leg_plot_reconst']  = False   # plot and compare the reconstructed distribution
    params['Leg_test_interp']   = False  # test interpolation of W2D by plotting
    params['plot_bcoeffs']      = True  # plot b-coefficients
    params['Leg_npts_r']        = 500   # number of radial points for plotting of the Legendre expansion
    params['Leg_npts_th']       = 360   # number of angular points for plotting of the Legendre expansion
        
    """ *** PES *** """
    params['pes_npts']       = 1000   # numer of points for PES evaluation
    params['pes_max_k']      = 3.0     # maximum momentum in a.u. Must be lower than the momentum range for W2D
    params['pes_lmax']       = 100

    """ *** PECD *** """
    params['pecd_lmax']       = 4               # maximum angular momentum in the spherical harmonics expansion of the momentum probability function
    params['pecd_momenta']    = [1.15]   # (a.u.) (list) at what values of the electron momentum do you want PECD?


    return params


"""
        Available functions:

            1) rho1Dr:  rho(r,theta0,t)     - radial density for fixed angle
            2) rho1Dth: rho(r0,theta,t)     - angular density for fixed distance
            3) rho2D:   rho(r,theta,phi0,t) - polar 2D map, fixed phi0
            4) rho2Dav: rho_av(r,theta,t)   - polar 2D map, averaged over phi
            5) rho3Dcart: rho3D(x,y,z,t)    - 4D plot with Mayavi


            1) W1Dth:  W(k0,theta,t)     - 1D momentum probability density in theta 
            2) W1Dk: W(k,theta0,t)      -   
            3) W2D:  W(k,theta,phi0,t)     - 2D momentum probability density in k,theta
            4) W2Dav: W(k,theta,t)     - 2D momentum probability density in k,theta, phi-averaged
            5) PECD
"""