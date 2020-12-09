#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
import numpy as np

class propagate():

    def __init__(self):
        pass

    def prop_wf(self,method,basis,ini_state,params):
        """ main method for propagating the wavefunction"""
        """ method (str):       'static' - solve time-independent Schrodinger equation at time t=t0
                                'direct' - propagate the wavefunction with direct exponentiation of the Hamiltonian
                                'krylov' - propagate the wavefunction with iterative method (Lanczos: Krylov subspace method)
            basis (str):        'prim' - use primitive basis
                                'adiab' - use adiabatic basis from t=t0
            ini_state (dict):   {'method': manual,file,calc 'filename':filename}
                                    'method': manual - place the initial state manually in the primitive basis
                                              file - read the inititial state from file given in primitive basis
                                              calc - calculate initial state from given orbitals, by projection onto the primitive basis (tbd)
                                    'filename': filename - filename containing the initial state
            params (dict):      dictionary with all relevant numerical parameters of the calculation: t0,tend,dt,lmin,lmax,nbins,nlobatto,binwidth,tolerances
            potential (str):    name of the potential energy function (for now it is in an analytic form)
            field (str):        name of the electric field function used (it can further read from file or return analytic field)
        """



if __name__ == "__main__":      


    params = {}
    """==== propagation parameters ===="""
    params['t0'] = 0.0
    params['tend'] = 0.0
    params['dt'] = 1.0

    
    """====basis set parameters===="""
    
    params['nlobatto'] = 4
    params['nbins'] = 2
    params['binwidth'] = 1.0
    params['rshift'] = 0.01

    params['lmin'] = 0
    params['lmax'] = 1
    
    """====runtime controls===="""

    method = "static"

    ini_state = "manual"

    basis = "prim"

    potential = "hydrogen"

    field = "field.txt"