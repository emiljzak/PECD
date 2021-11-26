#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
from textwrap import indent
from h5py._hl import datatype
import numpy as np
from scipy import sparse
from scipy.fftpack import fftn
from scipy.sparse.linalg import expm, expm_multiply, eigsh
from scipy.special import sph_harm
from scipy.special import eval_legendre

import quaternionic
import spherical
from sympy.core.numbers import Integer

from sympy.physics.wigner import gaunt
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.quantum.cg import CG
from sympy import N

import itertools
import json
import h5py

import unittest 

import MAPPING
import GRID
import BOUND
import CONSTANTS
import FIELD
import PLOTS
import ROTDENS

import time
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker

#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput
#from pycallgraph import Config



def prop_wf( params, ham0, psi0, maparray, Gr, euler, ieuler ):

    time_to_au      = CONSTANTS.time_to_au[ params['time_units'] ]
    wfn_saverate    = params['wfn_saverate']
 
    #rho =  sparse.csc_matrix(ham_init).getnnz() / np.prod(sparse.csc_matrix(ham_init).shape)
    #print("density of the sparse hamiltonian matrix = " + str(rho) )

    #rho =  ham1.getnnz() / np.prod(ham1.shape)
    #print("density of the sparse hamiltonian matrix after filter = " + str(rho) )
    #exit()
    #plt.spy(ham1,precision=params['sph_quad_tol'], markersize=2)
    #plt.show()
    

    #BOUND.plot_mat(ham_init.todense())
    #plt.spy(ham_init, precision=params['sph_quad_tol'], markersize=5)
    #plt.show()
    #ham0 /= 2.0


    Nbas0 = len(psi0)
    print("Nbas for the bound Hamiltonian = " + str(Nbas0))

    print("Setting up time-grid")
    tgrid = np.linspace(    params['t0'] * time_to_au, 
                            params['tmax'] * time_to_au, 
                            int((params['tmax']-params['t0'])/params['dt']+1), 
                            endpoint = True )
    dt = params['dt'] * time_to_au

    print("Allocating wavepacket")

    if params['field_type']['function_name'] == "fieldRCPL":
        helicity = "R"
    elif params['field_type']['function_name'] == "fieldLCPL":
        helicity = "L"  
    elif params['field_type']['function_name'] == "fieldLP":
        helicity = "0"
    else:
        raise ValueError("Incorect field name")


    if params['wavepacket_format'] == "dat":
        flwavepacket      = open(   params['job_directory'] + 
                                    params['wavepacket_file'] +
                                    helicity + "_" + str(ieuler) 
                                    + ".dat", 'w' )

    elif params['wavepacket_format'] == "h5":

        flwavepacket =  h5py.File(  params['job_directory'] +
                                    params['wavepacket_file'] + 
                                    helicity + "_" + str(ieuler) + ".h5",
                                    mode='w')

    else:
        raise ValueError("incorrect/not implemented format")

  
    # Project the bound Hamiltonian onto the propagation Hamiltonian
    Nbas, psi_init  = PROJECT_PSI_GLOBAL(params,maparray,psi0) 
    ham_init        = PROJECT_HAM_GLOBAL(params, maparray, Nbas, Gr, ham0 )

    wavepacket        = np.zeros( ( len(tgrid), Nbas ) , dtype = complex )
    psi               = psi_init[:]
    psi[:]           /= np.sqrt( np.sum( np.conj(psi) * psi ) )



    if params['calc_free_energy'] == True:
        felenfile = open( params['job_directory'] + "FEL_energy.dat", 'w' )

    print("initialize electric field")
    Elfield = FIELD.Field(params)
    Fvec = Elfield.gen_field(tgrid) 
    
    if params['plot_elfield'] == True:
        PLOTS.plot_elfield(Fvec,tgrid,time_to_au)

    print(" Initialize the interaction matrix ")


    start_time = time.time()
    intmat0 = []
    h1, h2, h3 =  calc_intmat( maparray, Gr, Nbas) 
    intmat0.append(h1)
    intmat0.append(h2)
    intmat0.append(h3)
    end_time = time.time()
    print("time for calculation of dipole interaction matrix =  " + str("%10.3f"%(end_time-start_time)) + "s")



    Fvec = np.asarray(Fvec)
    Fvec = np.stack(( Fvec[i] for i in range(len(Fvec)) ), axis=1) 
    #Fvec += np.conjugate(Fvec)

    start_time_global = time.time()
    for itime, t in enumerate(tgrid): 

        start_time = time.time()
        if itime%10 == 0:
            print("t = " + str( "%10.1f"%(t/time_to_au)) + " as" + " normalization: " + str(np.sqrt( np.sum( np.conj(psi) * psi )) ) ) 
        else:
            print("t = " + str( "%10.1f"%(t/time_to_au)) + " as")
       
    
        #dip =   np.tensordot( Fvec[itime], intmat0, axes=([0],[2]) ) 
        #dip =   Elfield.gen_field(t)[0] * intmat0[:,:,0]  + Elfield.gen_field(t)[2] * intmat0[:,:,2]
        dip = Fvec[itime][0] * intmat0[0]  + Fvec[itime][1] * intmat0[1] + Fvec[itime][2] * intmat0[2]
        #print(Fvec[itime][1]* intmat0[1])
        #dip = sparse.csr_matrix(dip)
        #print("Is the full hamiltonian matrix symmetric? " + str(check_symmetric( ham0 + dip )))
                
        psi_out             = expm_multiply( -1.0j * ( ham_init + dip ) * dt, psi ) 
        wavepacket[itime,:] = psi_out
        psi                 = wavepacket[itime,:]

        end_time = time.time()
        print("time =  " + str("%10.3f"%(end_time-start_time)) + "s")

        if itime%wfn_saverate == 0:
            if params['wavepacket_format'] == "dat":
                flwavepacket.write( '{:10.3f}'.format(t) + 
                                    " ".join('{:15.5e}'.format(psi[i].real) + 
                                    '{:15.5e}'.format(psi[i].imag) for i in range(0,Nbas)) +
                                    '{:15.8f}'.format(np.sqrt(np.sum((psi[:].real)**2+(psi[:].imag)**2))) +
                                     "\n")

            elif params['wavepacket_format'] == "h5":

                flwavepacket.create_dataset(    name        = str('{:10.3f}'.format(t)), 
                                                data        = psi,
                                                dtype       = complex,
                                                compression = 'gzip' #no-loss compression. Compression with loss is possible and can save space.
                                            )
                
    end_time_global = time.time()
    print("The time for the wavefunction propagation is: " + str("%10.3f"%(end_time_global-start_time_global)) + "s")
    flwavepacket.close()

def PROJECT_HAM_GLOBAL(params, maparray, Nbas, Gr, ham0):

    ham = sparse.csr_matrix((Nbas, Nbas), dtype=complex) 
    
    # 0. Append the bound hamiltonian
    ham[:Nbas0,:Nbas0] = ham0

    #print(ham.todense())
    #plt.spy(ham,precision=1e-8, markersize=2)
    #plt.show()

    # 1. Build the full KEO in propagation space minus bound space
 
    start_time = time.time()
    keomat = BOUND.BUILD_KEOMAT_FAST( params, maparray, Nbas , Gr )
    end_time = time.time()
    print("Time for construction of KEO matrix in full propagation space is " +  str("%10.3f"%(end_time-start_time)) + "s")

    #plt.spy(keomat,precision=1e-8, markersize=2)
    #plt.show()

    #print("Shape of keomat: " + str(keomat.shape) )
    #print("Shape of ham: " + str(ham.shape) )

    keomat_copy = keomat.copy()
    keomat_copy += keomat.getH()
    for i in range(keomat.shape[0]):
        keomat_copy[i,i] /=2.0 #-= hmat.diagonal()[i]


    ham[:Nbas0,:Nbas0]  -= keomat_copy[:Nbas0, :Nbas0]
    #plt.spy(ham,precision=1e-4, markersize=2)
    #plt.show()

    ham                 += keomat_copy  

    #assert TEST_BOUNDARY_HAM(params,ham,Nbas0) == True, "Oh no! The bound Hamiltonian is incompatible with the full Hamiltonian."
    
    #plt.spy(ham,precision=1e-4, markersize=2)
    #plt.show()

    #BOUND.plot_mat(ham.todense())

    # 2. Optional: add "long-range potential" 
    # Build the full potential in propagation space minus bound spac
        # consider cut-offs for the electrostatic potential 
        #

    return ham


def PROJECT_PSI_GLOBAL(params, maparray, psi0):
    Nbas = len(maparray)
    Nbas0 = len(psi0)
    print("Nbas = " + str(Nbas) + ", Nbas0 = " + str(Nbas0))

    psi = np.zeros(Nbas, dtype = complex)    
    psi[:Nbas0] = psi0[:,params['ivec']]

    return Nbas, psi

def TEST_BOUNDARY_HAM(params,ham,Nbas0):
    """ diagonalize hmat """
    start_time = time.time()
    enr, coeffs = call_eigensolver(ham, params)
    end_time = time.time()
    print("Time for diagonalization of the full Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")


    """ diagonalize hmat0 """
    start_time = time.time()
    enr0, coeffs0 = call_eigensolver(ham[:Nbas0,:Nbas0], params)
    end_time = time.time()
    print("Time for diagonalization of the bound Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")

    print(enr-enr0)

    return np.allclose(enr,enr0,atol=1e-4)


def BUILD_HMAT0_ROT(params, Gr, maparray, Nbas, grid_euler, irun):
    """ Build the stationary hamiltonian with rotated ESP in unrotated basis, store the hamiltonian in a file """

    if params['read_ham_init_file'] == True:

        if params['hmat_format']   == "numpy_arr":
            if os.path.isfile(params['job_directory'] + params['file_hmat0'] + "_" + str(irun) + ".dat"  ):
        
                print (params['file_hmat0'] + " file exist")
                hmat = read_ham_init_rot(params,irun)
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
                raise ValueError("Incorrect file name for the Hamiltonian matrix")
                exit()

        elif params['hmat_format']   == "sparse_csr":

            if os.path.isfile(params['job_directory'] + params['file_hmat0'] + "_" + str(irun) + ".npz" ):
                print (params['file_hmat0'] + "_" + str(irun) + ".npz" + " file exist")
                ham0 =  read_ham_init_rot(params,irun)
                #plt.spy(ham0, precision=params['sph_quad_tol'], markersize=3, label="HMAT")
                #plt.legend()
                #plt.show()
        
                """ Alternatively we can read enr and coeffs from file"""

                """ diagonalize hmat """
                start_time = time.time()
                enr, coeffs = call_eigensolver(ham0, params)
                end_time = time.time()
                print("Time for diagonalization of field-free Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")

                return ham0, coeffs
            else:
                raise ValueError("Incorrect file name for the Hamiltonian matrix or file does not exists")
                exit()
    else:

        if params['hmat_format'] == 'numpy_arr':    
            hmat =  np.zeros((Nbas, Nbas), dtype=float)
        elif params['hmat_format'] == 'sparse_csr':
            if params['esp_mode']  == 'anton':
                hmat = sparse.csr_matrix((Nbas, Nbas), dtype=complex) #complex potential in Demekhin's work
            else:
                hmat = sparse.csr_matrix((Nbas, Nbas), dtype=complex) #if
        else:
            raise ValueError("Incorrect format type for the Hamiltonian")
            exit()

        """ calculate POTMAT """
        if params['esp_mode'] == "exact":
            """ Use Psi4 to generate values of the ESP at quadrature grid points. 
                Use jit for fast calculation of the matrix elements """
            potmat, potind = BOUND.BUILD_POTMAT0_ROT( params, maparray, Nbas, Gr, grid_euler, irun )   


        elif params['esp_mode'] == "multipoles":
            potmat, potind = BOUND.BUILD_POTMAT0_MULTIPOLES_ROT( params, maparray, Nbas , Gr, grid_euler, irun )
        
        elif params['esp_mode'] == "anton":
            potmat, potind = BOUND.BUILD_POTMAT0_ANTON_ROT( params, maparray, Nbas , Gr, grid_euler, irun )

        """ Put the indices and values back together in the Hamiltonian array"""
        for ielem, elem in enumerate(potmat):
            #print(elem[0])
            hmat[ potind[ielem][0], potind[ielem][1] ] = elem[0]


        #print("plot of hmat")

        #BOUND.plot_mat(hmat.todense())
        #plt.spy(hmat,precision=params['sph_quad_tol'], markersize=3, label="HMAT")
        #plt.legend()
        #plt.show()
        #exit()

        """ calculate KEO """
        start_time = time.time()
        #print(Gr.ravel())
        #exit()
        keomat = BOUND.BUILD_KEOMAT_FAST( params, maparray, Nbas , Gr )
        end_time = time.time()
        print("New implementation - time for construction of KEO matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")

        #start_time = time.time()
        #keomat = BOUND.BUILD_KEOMAT( params, maparray, Nbas , Gr )
        #end_time = time.time()
        #print("Old implementation - time for construction of KEO matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")
        
        hmat += keomat 
        #BOUND.plot_mat(hmat.todense())
        #print("plot of hmat")
        #BOUND.plot_mat(hmat)
        #plt.spy(hmat,precision=params['sph_quad_tol'], markersize=3, label="HMAT")
        #plt.legend()
        #plt.show()
        
        """ --- make the hamiltonian matrix hermitian --- """
        if params['hmat_format'] == 'numpy_arr':    
            ham0    = np.copy(hmat)
            ham0    += np.transpose(hmat.conjugate()) 
            for i in range(ham0.shape[0]):
                ham0[i,i] -= hmat.diagonal()[i]
            print("Is the field-free hamiltonian matrix symmetric? " + str(check_symmetric(ham0)))

        elif params['hmat_format'] == 'sparse_csr':
            #hmat = sparse.csr_matrix(hmat)
            hmat_csr_size = hmat.data.size/(1024**2)
            print('Size of the sparse Hamiltonian csr_matrix: '+ '%3.2f' %hmat_csr_size + ' MB')
            ham0 = hmat.copy()
            ham0 += hmat.getH()
            for i in range(ham0.shape[0]):
                ham0[i,i] /=2.0#-= hmat.diagonal()[i]
        else:
            raise ValueError("Incorrect format type for the Hamiltonian")
            exit()

        """ --- filter hamiltonian matrix  --- """

        if params['hmat_format'] == 'numpy_arr':    
            ham_filtered = np.where( np.abs(ham0) < params['hmat_filter'], 0.0, ham0)
            #ham_filtered = sparse.csr_matrix(ham_filtered)

        elif params['hmat_format'] == 'sparse_csr':
            nonzero_mask        = np.array(np.abs(ham0[ham0.nonzero()]) < params['hmat_filter'])[0]
            rows                = ham0.nonzero()[0][nonzero_mask]
            cols                = ham0.nonzero()[1][nonzero_mask]
            ham0[rows, cols]    = 0
            ham_filtered        = ham0.copy()


        #plt.spy(ham0, precision=params['sph_quad_tol'], markersize=3, label="HMAT")
        #plt.legend()
        #plt.show()
        #exit()
        #print("Maximum real part of the hamiltonian matrix = " + str(np.max(ham_filtered.real)))
        #print("Maximum imaginary part of the hamiltonian matrix = " + str(np.max(ham_filtered.imag)))
        #exit()

        if params['save_ham0'] == True:
            if params['hmat_format'] == 'sparse_csr':
                sparse.save_npz( params['job_directory']  + params['file_hmat0'] + "_" + str(irun) , ham_filtered , compressed = False )
            elif params['hmat_format'] == 'numpy_arr':
                with open( params['job_directory'] + params['file_hmat0']+ "_" + str(irun) , 'w') as hmatfile:   
                    np.savetxt(hmatfile, ham_filtered, fmt = '%10.4e')
            print("Hamiltonian matrix saved.")

        """ diagonalize hmat """
        if params['hmat_format'] == 'numpy_arr':    
            start_time = time.time()
            enr, coeffs = np.linalg.eigh(ham_filtered , UPLO = 'U')
            end_time = time.time()
        elif params['hmat_format'] == 'sparse_csr':
            start_time = time.time()
            enr, coeffs = call_eigensolver(ham_filtered, params)
            end_time = time.time()

   
        print("Time for diagonalization of field-free Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")


        print("Normalization of initial wavefunctions: ")
        for v in range(params['num_ini_vec']):
            print(str(v) + " " + str(np.sqrt( np.sum( np.conj(coeffs[:,v] ) * coeffs[:,v] ) )))


        if params['save_psi0'] == True:
            psifile = open(params['job_directory']  + params['file_psi0']+ "_"+str(irun), 'w')
            for ielem,elem in enumerate(maparray):
                psifile.write( " %5d"%elem[0] +  " %5d"%elem[1] + "  %5d"%elem[2] + \
                                " %5d"%elem[3] +  " %5d"%elem[4] + "\t" + \
                                "\t\t ".join('{:10.5e}'.format(coeffs[ielem,v]) for v in range(0,params['num_ini_vec'])) + "\n")

        if params['save_enr0'] == True:
            with open(params['job_directory'] + params['file_enr0']+ "_"+str(irun), "w") as energyfile:   
                np.savetxt( energyfile, enr * CONSTANTS.au_to_ev , fmt='%10.5f' )
    

        """ Plot initial orbitals """
        if params['plot_ini_orb'] == True:
            PLOTS.plot_initial_orbitals(params,maparray,coeffs)


        return ham_filtered, coeffs

def call_eigensolver(A,params):
    if params['ARPACK_enr_guess'] == None:
        print("No eigenvalue guess defined")
    else:
        params['ARPACK_enr_guess'] /= CONSTANTS.au_to_ev


    if params['ARPACK_which'] == 'LA':
        print("using which = LA option in ARPACK: changing sign of the Hamiltonian")
        
        #B = A.copy()
        #B = B-A.getH()
        #print(B.count_nonzero())
        #if not B.count_nonzero()==0:
        #    raise ValueError('expected symmetric or Hermitian matrix!')

        enr, coeffs = eigsh(    -1.0 * A, k = params['num_ini_vec'], 
                                which=params['ARPACK_which'] , 
                                sigma=params['ARPACK_enr_guess'],
                                return_eigenvectors=True, 
                                mode='normal', 
                                tol = params['ARPACK_tol'],
                                maxiter = params['ARPACK_maxiter'])
        enr *= -1.0
        enr = np.sort(enr)

        coeffs_sorted = np.copy(coeffs)

        for i in range(params['num_ini_vec']):
            coeffs_sorted[:,i] = coeffs[:,params['num_ini_vec']-i-1]
        coeffs = coeffs_sorted


    else:
        enr, coeffs = eigsh(    A, k = params['num_ini_vec'], 
                                which=params['ARPACK_which'] , 
                                sigma=params['ARPACK_enr_guess'],
                                return_eigenvectors=True, 
                                mode='normal', 
                                tol = params['ARPACK_tol'],
                                maxiter = params['ARPACK_maxiter'])

    print(coeffs.shape)
    print(enr.shape)
    #sort coeffs

    return enr, coeffs

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


def read_ham_init(params):
    if params['hmat_format'] == 'sparse_csr':
        hmat = sparse.load_npz( params['working_dir'] + params['file_hmat0']+ ".npz" )
    elif params['hmat_format'] == 'numpy_arr':
        with open( params['working_dir'] + params['file_hmat0'] , 'r') as hmatfile:   
            hmat = np.loadtxt(hmatfile)
    return hmat


def read_ham_init_rot(params,irun):
    #rotated version
    if params['hmat_format'] == 'sparse_csr':
        hmat = sparse.load_npz( params['job_directory'] + params['file_hmat0']+ "_" + str(irun) + ".npz" )
    elif params['hmat_format'] == 'numpy_arr':
        with open( params['job_directory'] + params['file_hmat0'] + "_"+str(irun) + ".dat", 'r') as hmatfile:   
            hmat = np.loadtxt(hmatfile)
    return hmat


def calc_intmat(maparray,rgrid,Nbas):  

    #field: (E_-1, E_0, E_1) in spherical tensor form
    """calculate the <Y_l'm'(theta,phi)| d(theta,phi) | Y_lm(theta,phi)> integral """

    if params['hmat_format'] == 'numpy_arr':    
        intmat =   np.zeros(( Nbas , Nbas ), dtype = complex)
    elif params['hmat_format'] == 'sparse_csr':
        intmat1 = sparse.csr_matrix(( Nbas, Nbas ), dtype = complex)
        intmat2 = sparse.csr_matrix(( Nbas, Nbas ), dtype = complex)
        intmat3 = sparse.csr_matrix(( Nbas, Nbas ), dtype = complex)

    D = np.zeros(3)

    """precompute all necessary 3-j symbols"""
    #generate arrays of 3j symbols with 'spherical':
    tjmat = gen_3j_dip(params['bound_lmax'])
  

    for i in range(Nbas):
        rin = rgrid[ maparray[i][0], maparray[i][1] -1 ]
        for j in range(Nbas):

            if  maparray[i][2] == maparray[j][2]:
                #D[0] = N( gaunt( maparray[i][3], 1, maparray[j][3], maparray[i][4], -1, maparray[j][4] ) )
                #D[1] = N( gaunt( maparray[i][3], 1, maparray[j][3], maparray[i][4], 0, maparray[j][4] ) ) * np.sqrt(2.)
                #D[2] = N( gaunt( maparray[i][3], 1, maparray[j][3], maparray[i][4], 1, maparray[j][4] ) ) 

                #intmat1[i,j] = np.sqrt( 2.0 * np.pi / 3.0 ) * D[0] * rin 
                #intmat2[i,j] = np.sqrt( 2.0 * np.pi / 3.0 ) * D[1] * rin 
                #intmat3[i,j] = np.sqrt( 2.0 * np.pi / 3.0 ) * D[2] * rin 

                D[:] = tjmat[ maparray[i][3], maparray[j][3], maparray[i][4]+maparray[i][3], : ] #-1, 0 , +1
                if maparray[j][4] == maparray[i][4]:
                    intmat2[i,j] = np.sqrt( 2.0 * np.pi / 3.0 ) * D[1] * rin * np.sqrt(2.)
                elif maparray[j][4] ==1- maparray[i][4]:
                    intmat1[i,j] = np.sqrt( 2.0 * np.pi / 3.0 ) * D[0] * rin 
                elif maparray[j][4] ==-1- maparray[i][4]:
                    intmat3[i,j] = np.sqrt( 2.0 * np.pi / 3.0 ) * D[2] * rin 


    #plt.spy(intmat_new1, precision=params['sph_quad_tol'], markersize=5)
    #plt.spy(intmat1, precision=params['sph_quad_tol'], markersize=5, color='r')
    #plt.spy(intmat2, precision=params['sph_quad_tol'], markersize=5, color='g')
    #plt.spy(intmat3, precision=params['sph_quad_tol'], markersize=5, color='b')
    #plt.show()
    #exit()
    #rtol=1e-05
    #atol=1e-08
    #print(intmat_new)

    #np.allclose(intmat, intmat_new, rtol=rtol, atol=atol)


    #intmat += np.conjugate(intmat.T)

    #print("Interaction matrix")
    #with np.printoptions(precision=4, suppress=True, formatter={'complex': '{:15.8f}'.format}, linewidth=400):
    #    print(intmat)
    #print("Is the interaction matrix symmetric? " + str(check_symmetric(intmat)))

    return intmat1,intmat2,intmat3


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.conj(), rtol=rtol, atol=atol)

def cart2sph(x,y,z):
    r=np.sqrt(x**2+y**2+z**2)
    theta=np.arctan(np.sqrt(x**2+y**2)/z)
    phi=np.arctan(y/x)
    return r,theta,phi
    
def spharmcart(l,m,x,y,z):
    tol = 1e-8
    if all(np.abs(x)>tol) and all(np.abs(z)>tol): 
        r       =   np.sqrt(x**2+y**2+z**2)
        theta   =   np.arctan(np.sqrt(x**2+y**2)/z)
        phi     =   np.arctan(y/x)
    else:
        r       = 0.0
        theta   = 0.0
        phi     = 0.0
    return sph_harm(m, l, phi, theta)



def gen_euler_grid_theta_chi(n_euler):
    alpha_1d        = list(np.linspace(0, 2*np.pi,  num=n_euler, endpoint=False))
    beta_1d         = list(np.linspace(0, np.pi,    num=n_euler, endpoint=True))
    gamma_1d        = list(np.linspace(0, 2*np.pi,  num=1, endpoint=False))
    euler_grid_3d   = np.array(list(itertools.product(*[alpha_1d, beta_1d, gamma_1d]))) #cartesian product of [alpha,beta,gamma]

    #we can choose meshgrid instead
    #euler_grid_3d_mesh = np.meshgrid(alpha_1d, beta_1d, gamma_1d)
    #print(euler_grid_3d_mesh[0].shape)
    #print(np.vstack(euler_grid_3d_mesh).reshape(3,-1).T)

    n_euler_3d      = euler_grid_3d.shape[0]
    print("\nTotal number of 3D-Euler grid points: ", n_euler_3d , " and the shape of the 3D grid array is:    ", euler_grid_3d.shape)
    #print(euler_grid_3d)
    return euler_grid_3d, n_euler_3d

def rotate_coefficients(ind_euler,coeffs,WDMATS,lmax,Nr):
    """ take coefficients and rotate them by angles = (alpha, beta, gamma) """
    #ind_euler - index of euler angles in global 3D grid

    Dsize = (lmax+1)**2 
    Dmat = np.zeros((Dsize,Dsize), dtype = complex)

    #fill up the D super-matrix
    ind_start = np.zeros(lmax+1, dtype=Integer)
    ind_end = np.zeros(lmax+1, dtype=Integer)

    ind_start[0] = 0
    ind_end[0] = 0

    for l in range(0,lmax):
        ind_start[l+1] = ind_start[l] +  2 * l +1
        ind_end[l+1] = ind_start[l+1] + 2 * (l+1) +1  #checked by hand 5 Jun 2021

    for l in range(0,lmax+1):
        #print(WDMATS[l][:,:,ind_euler].shape)
        #print(l, ind_start[l], ind_end[l])
        Dmat[ind_start[l]:ind_end[l],ind_start[l]:ind_end[l]] = WDMATS[l][:,:,ind_euler]

    #with np.printoptions(precision=4, suppress=True, formatter={'complex': '{:15.8f}'.format}, linewidth=400):
    #    print(Dmat)

    coeffs_rotated = np.zeros(coeffs.shape[0], dtype = complex)

    #print("number of radial basis points/functions: " + str(Nr))

    xi_start    = np.zeros(Nr, dtype=Integer)
    xi_end      = np.zeros(Nr, dtype=Integer)
    xi_start[0] = 0 
    xi_end[0]   = Dsize

    for xi in range(Nr-1):
        xi_start[xi+1] = xi_start[xi] + Dsize
        xi_end[xi+1]   = xi_end[xi] +  Dsize
        

    print("size of single D-mat block: " + str(-xi_start[1]+xi_end[1]))

    for xi in range(Nr):
        #print(xi, xi_start[xi], xi_end[xi]) #checked by hand 5 Jun 2021
        coeffs_rotated[xi_start[xi]:xi_end[xi]] = np.matmul(Dmat,coeffs[xi_start[xi]:xi_end[xi]])

    #print(coeffs_rotated)

    return coeffs_rotated



def gen_3j_dip(lmax):
    """precompute all necessary 3-j symbols for dipole matrix elements"""
    #store in arrays:
    # 2) tjmat[l,l',m,sigma] = [0,...lmax,0...lmax,0,...,m+l,0...2]
    tjmat = np.zeros( (lmax+1, lmax+1, 2*lmax+1, 3), dtype = float)

    for mu in range(0,3):
        for l1 in range(lmax+1):
            for l2 in range(lmax+1):
                for m in range(-l1,l1+1):
                    
                    tjmat[l1,l2,l1+m,mu] = spherical.Wigner3j(l1, 1, l2, m, mu-1, -(m+mu-1)) * spherical.Wigner3j(l1, 1, l2, 0, 0, 0)
                    tjmat[l1,l2,l1+m,mu] *= np.sqrt((2*float(l1)+1) * (2.0*float(1.0)+1) * (2*float(l2)+1)/(4.0*np.pi)) * (-1)**(m+mu-1)

    #print("3j symbols in array:")
    #print(tjmat)
    return tjmat


def read_euler_grid():   
    with open( "grid_euler.dat" , 'r') as eulerfile:   
        grid_euler = np.loadtxt(eulerfile)
    return grid_euler

def save_map(map,file):
    fl = open(file,'w')
    for elem in map:   
        fl.write(" ".join('{:5d}'.format(elem[i]) for i in range(0,6)) + "\n")
    fl.close()


if __name__ == "__main__":   

    start_time_total = time.time()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    print(" ")
    print("---------------------- START PROPAGATE --------------------")
    print(" ")
    
    ibatch = int(sys.argv[1]) # id of batch of Euler angles grid run
    os.chdir(sys.argv[2])
    path = os.getcwd()

    print("dir: " + path)

    with open('input_prop', 'r') as input_file:
        params = json.load(input_file)

    print(" ")
    print("---------------------- INPUT ECHO --------------------")
    print(" ")

    for key, value in params.items():
        print(key, ":", value)

    maparray0, Nbas0 = MAPPING.GENMAP_FEMLIST(  params['FEMLIST'],
                                                            params['bound_lmax'],
                                                            params['map_type'],
                                                            params['job_directory'] )


    maparray, Nbas = MAPPING.GENMAP_FEMLIST(  params['FEMLIST_PROP'],
                                                            params['bound_lmax'],
                                                            params['map_type'],
                                                            params['job_directory'] )

    save_map(maparray0,params['job_directory'] + 'map0.dat')
    save_map(maparray,params['job_directory'] + 'map_global.dat')

    Gr0, Nr0                       = GRID.r_grid(             params['bound_nlobs'], 
                                                            params['bound_nbins'] , 
                                                            params['bound_binw'],  
                                                            params['bound_rshift'] )

    Gr, Nr                       = GRID.r_grid(             params['bound_nlobs'], 
                                                            params['prop_nbins'] , 
                                                            params['bound_binw'],  
                                                            params['bound_rshift'] )

    """ Read grid of Euler angles"""
    grid_euler  = read_euler_grid()

    grid_euler = grid_euler.reshape(-1,3)     

    N_Euler = grid_euler.shape[0]

    N_per_batch = int(N_Euler/params['N_batches'])

    maparray_chi, Nbas_chi = MAPPING.GENMAP_FEMLIST( params['FEMLIST'],  0, \
                                params['map_type'], path )

    save_map(maparray_chi,params['job_directory'] + 'map_chi.dat')

    for irun in range(ibatch * N_per_batch, (ibatch+1) * N_per_batch):

        #print(grid_euler[irun])
        """ Generate Initial Hamiltonian with rotated electrostatic potential in unrotated basis """
        ham0, psi0 = BUILD_HMAT0_ROT(params, Gr0, maparray0, Nbas0, grid_euler, irun)

        prop_wf(params, ham0, psi0, maparray, Gr, grid_euler[irun], irun)


end_time_total = time.time()
print("Global time =  " + str("%10.3f"%(end_time_total-start_time_total)) + "s")
