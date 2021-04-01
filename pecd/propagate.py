#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
import sys





def prop_wf(self,params,psi0):
    """ main method for propagating the wavefunction"""

    """params (dict):      dictionary with all relevant numerical parameters of the calculation: t0,tend,dt,lmin,lmax,nbins,nlobatto,binwidth,tolerances,etc."""
    """ psi0 (list): spectral representation of the initial wavefunction """
    """ method (str):       'static' - solve time-independent Schrodinger equation at time t=t0
                            'dynamic_direct' - propagate the wavefunction with direct exponentiation of the Hamiltonian
                            'dynamic_lanczos' - propagate the wavefunction with iterative method (Lanczos: Krylov subspace method)
        basis (str):        'prim' - use primitive basis
                            'adiab' - use adiabatic basis from t=t0
        ini_state (dict):   {'method': manual,file,calc 'filename':filename}
                                'method': manual - place the initial state manually in the primitive basis
                                            file - read the inititial state from file given in primitive basis
                                            calc - calculate initial state from given orbitals, by projection onto the primitive basis (tbd)
                                'filename': filename - filename containing the initial state

        potential (str):    name of the potential energy function (for now it is in an analytic form)
        field_type (str):   type of field: analytic or numerical
        field (str):        name of the electric field function used (it can further read from file or return analytic field)
        scheme (str):       quadrature scheme used in angular integration
    """



if __name__ == "__main__":      

    params = input.gen_input()


