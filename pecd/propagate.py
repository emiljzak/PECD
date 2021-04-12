#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
import sys
import MAPPING
import input


def prop_wf(params,psi0):
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

def read_coeffs(filename,nvecs):

    coeffs = []
    fl = open( filename , 'r' )
    for line in fl:
        words   = line.split()
        i       = int(words[0])
        n       = int(words[1])
        xi      = int(words[2])
        l       = int(words[3])
        m       = int(words[4])
        c       = []
        for ivec in range(nvecs):
            c.append(float(words[5+ivec]))
        coeffs.append([i,n,xi,l,m,np.asarray(c)])
    return coeffs

def proj_wf0_wfinit_dvr(coeffs0, marray, Nbas_global):
    psi = []
    for ivec in range(Nbas_global):
        if ivec < len(coeffs0):
            psi.append([marray[ivec][0],marray[ivec][1],marray[ivec][2],\
                        marray[ivec][3],marray[ivec][4],coeffs0[ivec][5]])
        else:
            psi.append([marray[ivec][0],marray[ivec][1],marray[ivec][2],\
                        marray[ivec][3],marray[ivec][4],0.0])
    """ for ivec in range(Nbas_global):
        for icoeffs0 in coeffs:
            if  marray[ivec][0] == icoeffs0[0]   and \
                marray[ivec][1] == icoeffs0[1]   and \
                marray[ivec][3] == icoeffs0[3]   and \
                marray[ivec][4] == icoeffs0[4]: #i,n,l,m
                    psi.append([marray[ivec][0],marray[ivec][1],marray[ivec][2],\
                                marray[ivec][3],marray[ivec][4],icoeffs[5]])
                break

        psi.append([marray[ivec][0],marray[ivec][1],marray[ivec][2],\
                    marray[ivec][3],marray[ivec][4],icoeffs[5]])
    """
    return psi
if __name__ == "__main__":      

    params = input.gen_input()


    maparray_global, Nbas_global = MAPPING.GENMAP_FEMLIST(  params['FEMLIST'],
                                                            params['bound_lmax'],
                                                            params['map_type'],
                                                            params['working_dir'] )

    coeffs0 = read_coeffs( params['working_dir'] + params['file_psi0'], 1 )


    psi_init = proj_wf0_wfinit_dvr(coeffs0, maparray_global, Nbas_global)
    print(psi_init)