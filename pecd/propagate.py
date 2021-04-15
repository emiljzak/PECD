#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from sympy.physics.wigner import gaunt
from sympy import N

import MAPPING
import input
import GRID
import BOUND
import CONSTANTS
import FIELD
import PLOTS

import time
import os
import sys

import matplotlib.pyplot as plt


def prop_wf( params, ham_init, psi_init, maparray, Gr ):


    time_to_au = CONSTANTS.time_to_au[ params['time_units'] ]

    ham0 = ham_init.copy()
    ham0 += np.transpose(ham_init) 
    ham0 /= 2.0


    psi_init_t = psi_init.copy()
    psi_init_t = np.transpose(psi_init_t)

    print( np.dot( psi_init_t, np.dot(ham0,psi_init) ))

    Nbas = len(psi_init)
    print("Nbas = " + str(Nbas))

    print("Setting up time-grid")
    tgrid = np.linspace(    params['t0'] * time_to_au, 
                            params['tmax'] * time_to_au, 
                            int((params['tmax']-params['t0'])/params['dt']+1), 
                            endpoint = True )
    dt = params['dt'] * time_to_au

    print("Allocating wavepacket")
    flwavepacket      = open( params['wavepacket_file'],'w' )
    wavepacket        = np.zeros( ( len(tgrid), Nbas ) , dtype=complex )
    psi               = np.zeros( Nbas, dtype=complex ) 
    psi               = psi_init
    psi[:]           /= np.sqrt( np.sum( np.conj(psi) * psi ) )


    print(" Initialize the interaction matrix ")
    intmat  = np.zeros(( Nbas , Nbas ), dtype = complex)
    intmat0 = np.zeros(( Nbas , Nbas, 3 ), dtype=complex)

    intmat0[:,:,0] = calc_intmat( [1.0, 0.0, 0.0], maparray, Gr, Nbas)  
    intmat0[:,:,1] = calc_intmat( [0.0, 1.0, 0.0], maparray, Gr, Nbas)  
    intmat0[:,:,2] = calc_intmat( [0.0, 0.0, 1.0], maparray, Gr, Nbas)  


    print("initialize electric field")
    Elfield = FIELD.Field(params)
    
    if params['plot_elfield'] == True:
        Fvec = Elfield.gen_field(tgrid)
        fig = plt.figure()
        ax = plt.axes()
        ax.scatter(tgrid,Fvec[0].real)
        ax.scatter(tgrid,Fvec[0].imag)
        ax.scatter(tgrid,Fvec[2])
        plt.show()


    start_time_global = time.time()
    for itime, t in enumerate(tgrid): 
        print("t = " + str( "%10.1f"%(t/time_to_au)) + " as" + " normalization: " + str(np.sqrt( np.sum( np.conj(psi) * psi )) ) ) 
       
        UMAT                = linalg.expm( -1.0j * ( ham0 + 0.0 * np.tensordot( Elfield.gen_field(t), intmat0, axes=([0],[2]) ) ) * dt ) 
        wavepacket[itime,:] = np.dot( UMAT , psi )
        psi                 = wavepacket[itime,:]

        flwavepacket.write( '{:10.3f}'.format(t) + 
                            " ".join('{:16.8e}'.format(psi[i].real) + '{:16.8e}'.format(psi[i].imag) for i in range(0,Nbas)) +\
                            '{:15.8f}'.format(np.sqrt(np.sum((psi[:].real)**2+(psi[:].imag)**2))) + "\n")
        
    end_time_global = time.time()
    print("The time for the wavefunction propagation is: " + str("%10.3f"%(end_time_global-start_time_global)) + "s")

    print("=====================================")
    print("==post-processing of the wavepacket==")
    print("====================================="+"\n")


    print("=========")
    print("==Plots==")
    print("=========")
        
    plot_times = []
    for index,item in enumerate(params['plot_controls']["plottimes"]):

        if int( item * time_to_au / dt) > len(tgrid):
            print("removing time: " + str(item) + " from plotting list. Time exceeds the propagation time-grid!")
        else:
            plot_times.append(item * time_to_au/dt)
    print("Final list of plottime indices in tgrid:")
    print(plot_times)

    #merge with maparray
    maparray = np.asarray(maparray)

    for itime, t in enumerate(tgrid): 
        for ielem in plot_times:
            if itime == ielem:
                psi[:] = wavepacket[itime,:]
                
                PLOTS.plot_snapshots(params,psi,maparray,Gr)




def BUILD_HMAT(params,maparray,Nbas,ham0):

    if params['read_ham_init_file'] == True and os.path.isfile(params['working_dir'] + params['file_hmat_init'] ):
        
        print (params['file_hmat_init'] + " file exist")
        hmat = read_ham_init(params)

        """ diagonalize hmat """
        start_time = time.time()
        enr, coeffs = np.linalg.eigh(hmat, UPLO = 'U')
        end_time = time.time()
        print("Time for diagonalization of field-free Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")

        #BOUND.plot_mat(hmat)
        #plt.spy(hmat,precision=params['sph_quad_tol'], markersize=2)
        #plt.show()
        return hmat, coeffs
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
        sparse.load_npz( params['working_dir'] + params['file_hmat_init'] , hmat , compressed = False )
    elif params['hmat_format'] == 'regular':
        with open( params['working_dir'] + params['file_hmat_init'] , 'r') as hmatfile:   
            hmat = np.loadtxt(hmatfile)
    return hmat


def calc_intmat(field,maparray,rgrid,Nbas):  

    #field: (E_-1, E_0, E_1) in spherical tensor form
    """calculate the <Y_l'm'(theta,phi)| d(theta,phi) | Y_lm(theta,phi)> integral """
    intmat =   np.zeros(( Nbas , Nbas ), dtype = complex)

    D = np.zeros(3)
    for i in range(Nbas):
        rin = rgrid[ maparray[i][0], maparray[i][1] ]
        for j in range(Nbas):
            if  maparray[i][2] == maparray[j][2]:
                D[0] = N( gaunt( maparray[i][3], 1, maparray[j][3], maparray[i][4], -1, maparray[j][4] ) )
                D[1] = N( gaunt( maparray[i][3], 1, maparray[j][3], maparray[i][4], 0, maparray[j][4] ) )
                D[2] = N( gaunt( maparray[i][3], 1, maparray[j][3], maparray[i][4], 1, maparray[j][4] ) )

                intmat[i,j] = np.sqrt( 2.0 * np.pi / 3.0 ) * np.dot(field, D) * rin 


    intmat += np.conjugate(intmat.T)

    #print("Interaction matrix")
    #with np.printoptions(precision=4, suppress=True, formatter={'complex': '{:15.8f}'.format}, linewidth=400):
    #    print(intmat)
    print("Is the interaction matrix symmetric? " + str(check_symmetric(intmat)))
    return intmat


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, np.conjugate(a).T, rtol=rtol, atol=atol)

if __name__ == "__main__":      

    params = input.gen_input()


    maparray_global, Nbas_global = MAPPING.GENMAP_FEMLIST(  params['FEMLIST'],
                                                            params['bound_lmax'],
                                                            params['map_type'],
                                                            params['working_dir'] )

    Gr, Nr = GRID.r_grid(params['bound_nlobs'] , 
                        params['bound_nbins'] + params['nbins'], 
                        params['bound_binw'],  
                        params['bound_rshift'] )

    coeffs0 = read_coeffs( params['working_dir'] + params['file_psi0'], 1 )


    psi_init = proj_wf0_wfinit_dvr(coeffs0, maparray_global, Nbas_global)
    #print(psi_init)

    ham0 = read_ham0(params)
    #print(ham0)
    #plt.spy(ham0,precision=params['sph_quad_tol'], markersize=5)
    #plt.show()

    ham_init, psi_init = BUILD_HMAT(params,maparray_global, Nbas_global,ham0)

    prop_wf(params,ham_init,psi_init[:,params['ivec']],maparray_global, Gr)