#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
from scipy import sparse

import MAPPING
import input
import GRID
import BOUND
import CONSTANTS

import time
import os
import sys

import matplotlib.pyplot as plt

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


def BUILD_HMAT(params,maparray,Nbas,ham0):

    if params['read_ham_init_file'] == True and os.path.isfile(params['working_dir'] + params['file_hmat'] ):
        print (params['file_hmat'] + " file exist")
        hmat = read_ham_init(params)
        BOUND.plot_mat(hmat)
        plt.spy(hmat,precision=params['sph_quad_tol'], markersize=2)
        plt.show()
        return hmat
    else:
        #r_grid is for now only implemented for single bin width. We need to improve that
        Gr, Nr = GRID.r_grid(params['bound_nlobs'] , 
                            params['bound_nbins'] + params['nbins'], 
                            params['bound_binw'],  
                            params['bound_rshift'] )

        if params['hmat_format'] == 'csr':
            hmat = sparse.csr_matrix((Nbas, Nbas), dtype=np.float64)
        elif params['hmat_format'] == 'regular':
            hmat = np.zeros((Nbas, Nbas), dtype=np.float64)

        """ calculate hmat """
        potmat, potind = BOUND.BUILD_POTMAT0( params, maparray, Nbas , Gr )      
        for ielem, elem in enumerate(potmat):
            #print(potind[ielem][0],potind[ielem][1])
            hmat[ potind[ielem][0],potind[ielem][1] ] = elem[0]
        
        start_time = time.time()
        keomat = BOUND.BUILD_KEOMAT0( params, maparray, Nbas , Gr )
        end_time = time.time()
        print("Time for construction of KEO matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")
        
        hmat += keomat 

        print("plot of hmat")
        #BOUND.plot_mat(hmat)
        plt.spy(hmat,precision=params['sph_quad_tol'], markersize=2)
        plt.show()
        
        """ diagonalize hmat """
        start_time = time.time()
        enr, coeffs = np.linalg.eigh(hmat, UPLO = 'U')
        end_time = time.time()
        print("Time for diagonalization of field-free Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")


        #BOUND.plot_wf_rad(0.0, params['bound_binw'],1000,coeffs,maparray,Gr,params['bound_nlobs'], params['bound_nbins']+params['nbins'])

        print("Normalization of the wavefunction: ")
        for v in range(params['num_ini_vec']):
            print(str(v) + " " + str(np.sqrt( np.sum( np.conj(coeffs[:,v] ) * coeffs[:,v] ) )))


        if params['save_ham_init'] == True:
            if params['hmat_format'] == 'csr':
                sparse.save_npz( params['working_dir'] + params['file_hmat_init'] , hmat , compressed = False )
            elif params['hmat_format'] == 'regular':
                with open( params['working_dir'] + params['file_hmat_init'] , 'w') as hmatfile:   
                    np.savetxt(hmatfile, hmat, fmt = '%10.4e')

        if params['save_psi_init'] == True:
            psifile = open(params['working_dir'] + params['file_psi_init'], 'w')
            for ielem,elem in enumerate(maparray):
                psifile.write( " %5d"%elem[0] +  " %5d"%elem[1] + "  %5d"%elem[2] + \
                                " %5d"%elem[3] +  " %5d"%elem[4] + "\t" + \
                                "\t\t ".join('{:10.5e}'.format(coeffs[ielem,v]) for v in range(0,params['num_ini_vec'])) + "\n")

        if params['save_enr_init'] == True:
            with open(params['working_dir'] + params['file_enr_init'], "w") as energyfile:   
                np.savetxt( energyfile, enr * CONSTANTS.ev_to_au , fmt='%10.5f' )
    

        return hmat, coeffs

""" ============ KEOMAT ============ """
def BUILD_KEOMAT( params, maparray, Nbas, Gr ):
    nlobs = params['nlobs']
    """call Gauss-Lobatto rule """
    x   =   np.zeros(nlobs)
    w   =   np.zeros(nlobs)
    x,w =   GRID.gauss_lobatto(nlobs,14)
    x   =   np.array(x)
    w   =   np.array(w)

    keomat =  np.zeros((Nbas, Nbas), dtype=np.float64)

    for i in range(Nbas):
        rin = Gr[maparray[i][0],maparray[i][1]]
        for j in range(i,Nbas):
            if maparray[i][3] == maparray[j][3] and maparray[i][4] == maparray[j][4]:
                keomat[i,j] = calc_keomatel(maparray[i][0], maparray[i][1],\
                                            maparray[i][3], maparray[j][0], maparray[j][1], x, w, rin, \
                                            params['bound_rshift'],params['bound_binw'])

    #print("KEO matrix")
    #with np.printoptions(precision=3, suppress=True, formatter={'float': '{:10.3f}'.format}, linewidth=400):
    #    print(0.5*keomat)

    #plt.spy(keomat, precision=params['sph_quad_tol'], markersize=5)
    #plt.show()

    return  0.5 * keomat

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

def read_ham0(params):
    if params['hmat_format'] == 'csr':
        sparse.load_npz( params['working_dir'] + params['file_hmat0'] , hmat0 , compressed = False )
    elif params['hmat_format'] == 'regular':
        with open( params['working_dir'] + params['file_hmat0'] , 'r') as hmatfile:   
            hmat0 = np.loadtxt(hmatfile)
    return hmat0

def read_ham_init(params):
    if params['hmat_format'] == 'csr':
        sparse.load_npz( params['working_dir'] + params['file_hmat'] , hmat , compressed = False )
    elif params['hmat_format'] == 'regular':
        with open( params['working_dir'] + params['file_hmat'] , 'r') as hmatfile:   
            hmat = np.loadtxt(hmatfile)
    return hmat


if __name__ == "__main__":      

    params = input.gen_input()


    maparray_global, Nbas_global = MAPPING.GENMAP_FEMLIST(  params['FEMLIST'],
                                                            params['bound_lmax'],
                                                            params['map_type'],
                                                            params['working_dir'] )

    coeffs0 = read_coeffs( params['working_dir'] + params['file_psi0'], 1 )


    psi_init = proj_wf0_wfinit_dvr(coeffs0, maparray_global, Nbas_global)
    #print(psi_init)

    ham0 = read_ham0(params)
    #print(ham0)
    #plt.spy(ham0,precision=params['sph_quad_tol'], markersize=5)
    #plt.show()

    ham = BUILD_HMAT(params,maparray_global, Nbas_global,ham0)
