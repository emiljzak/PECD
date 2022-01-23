#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2022 Emil Zak <emil.zak@cfel.de>, <emil.j.zak@gmail.com>
#
from textwrap import indent
from h5py._hl import datatype
import numpy as np
from scipy import sparse
from scipy.fftpack import fftn
from scipy.sparse.linalg import expm, expm_multiply, eigsh
from scipy.special import sph_harm
from scipy.special import eval_legendre

#import quaternionic
#import spherical
from sympy.core.numbers import Integer

from sympy.physics.wigner import gaunt
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.quantum.cg import CG
from sympy import N

import itertools
import json
import h5py

import unittest 

import wavefunction
import hamiltonian
import field
import plots
import constants

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
"""Molecular frame embedding

Args:
    arg
        String defining molecular frame

        .. code-block:: python

            water.frame = "diag(inertia)" # rotates to a frame where the inertia tensor
                                            # becomes diagonal
            water.frame = "zxy" # permutes the x, y, and z axes
            water.pol = [[9.1369, 0, 0], [0, 9.8701, 0], [0, 0, 9.4486]]
            water.frame = "pol" # rotates frame with water.pol matrix
            water.frame = "diag(pol)" # rotates to a frame where water.pol tensor
                                        # becomes diagonal
            water.frame = "None" # or None, resets frame to the one defined
                                    # by the input  molecular geometry

Returns:
    array (3,3)
        Frame rotation matrix
"""


class Propagator():
    """Class contains methods related to the time-propagation of the wavefunction.

    Args:
        filename : str
           fdsfsd
    Kwargs:
        thresh : float
            fdsf

    Attrs:
        params : dict
            Keeps relevant parameters.
        irun : int
            id of the run over the Euler grid
    """

    def __init__(self,params,irun):

        self.params             = params
        self.irun               = irun
        self.time_to_au         = constants.time_to_au[ params['time_units'] ]
        self.wfn_saverate       = params['wfn_saverate']
    
    @staticmethod
    def calc_mat_density(mat):
        dens =  sparse.csc_matrix(mat).getnnz() / np.prod(sparse.csc_matrix(mat.shape))
        print("The density of the sparse matrix = " + str(dens) )

    #def plot_mat(self,mat):
    #    bound.plot_mat(mat.todense())

    def spy_mat(self,mat):
        plt.spy(mat, precision=self.params['sph_quad_tol'], markersize=5)
        plt.show()

    def pull_helicity(self):
        if self.params['field_type']['function_name'] == "fieldRCPL":
            helicity = "R"
        elif self.params['field_type']['function_name'] == "fieldLCPL":
            helicity = "L"  
        elif self.params['field_type']['function_name'] == "fieldLP":
            helicity = "0"
        else:
            raise ValueError("Incorect field name")

        return helicity

    def allocate_wavepacket(self,helicity):
        
        if self.params['wavepacket_format'] == "dat":

            flwavepacket      = open(   self.params['job_directory'] + 
                                        self.params['wavepacket_file'] +
                                        helicity + "_" + str(self.irun) 
                                        + ".dat", 'w' )

        elif self.params['wavepacket_format'] == "h5":

            flwavepacket =  h5py.File(  self.params['job_directory'] +
                                        self.params['wavepacket_file'] + 
                                        helicity + "_" + str(self.irun) + ".h5",
                                        mode = 'w')

        else:
            raise ValueError("incorrect/not implemented format for the wavepacket")

        return flwavepacket


    def gen_timegrid(self,ijifds):

        """ 
        Calculates equidistant time-grid for the wavefunction proapgation.
        Start and end points are included in the grid. 

        Returns: tuple
            tgrid: numpy 1D array
                time grid array
            dt: float
                time step

        Examples:
            If t0=0.0 and tmax = 10.0 and dt = 1.0 the grid has 11 elements:
            [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
        """


        tgrid = np.linspace(    self.params['t0'] * self.time_to_au, 
                                self.params['tmax'] * self.time_to_au, 
                                int((self.params['tmax']-self.params['t0'])/self.params['dt']+1), 
                                endpoint = True )
        dt    = self.params['dt'] * self.time_to_au
        
        return tgrid, dt

    def normalize_psi(self,psi):
        return psi/np.sqrt( np.sum( np.conj(psi) * psi ) )
 

    def prop_wf(self, ham_init, psi_init, intmat ):

        """ This function propagates the wavefunction with the time-dependent Hamiltonian and saves the wavepacket in a file.
        
        
        """

        self.calc_mat_density(ham_init)
        self.spy_mat(ham_init)
        self.plot_mat(ham_init)

        print("\n")
        print("Setting up time-grid...")
        print("\n")

        tgrid, dt = self.gen_timegrid()

        print("\n")
        print("Allocating wavepacket...")
        print("\n")

        helicity = self.pull_helicity()
        fl_wavepacket = self.allocate_wavepacket(helicity)

        print("\n")
        print("Normalizing initial wavepacket...")
        print("\n")

        psi = self.normalize_psi(psi_init)

        print("\n")
        print("Generating electric fields...")
        print("\n")

        FieldObj = field.Field(self.params)
        Fvec    = FieldObj.gen_field(tgrid) 
        
        if params['plot_elfield'] == True:
            plots.plot_elfield(Fvec,tgrid,self.time_to_au)



        start_time_global = time.time()
        for itime, t in enumerate(tgrid): 

            start_time = time.time()

            if itime%10 == 0:
                print("t = " + str( "%10.1f"%(t/self.time_to_au)) + " as" + " normalization: " + str(np.sqrt( np.sum( np.conj(psi) * psi )) ) ) 
            else:
                print("t = " + str( "%10.1f"%(t/self.time_to_au)) + " as")
        
            # building the electric dipole interaction matrix
            dip = Fvec[itime,0] * intmat - Fvec[itime,2] * intmat_hc #+ Fvec[itime,0] * intmat
           
            #dip = sparse.csr_matrix(dip)
            #print("Is the full hamiltonian matrix symmetric? " + str(check_symmetric(ham_init.todense() + dip.todense() )))
            #exit()
            
            #Note: we can use multiple time-points with expm_multiply and ham_init as linear operator. Also action on a collection of vectors is possible.
            psi = expm_multiply( -1.0j * ( ham_init + dip  ) * dt, psi ) 

            #psi_out             = expm_multiply( -1.0j * ( ham_init + dip ) * dt, psi ) 
            #wavepacket[itime,:] = psi_out
            #psi                 = wavepacket[itime,:]


            if itime%wfn_saverate == 0:

                if self.params['wavepacket_format'] == "dat":
                    
                    fl_wavepacket.write(    '{:10.3f}'.format(t) + 
                                            " ".join('{:15.5e}'.format(psi[i].real) + 
                                            '{:15.5e}'.format(psi[i].imag) for i in range(0,Nbas)) +
                                            '{:15.8f}'.format(np.sqrt(np.sum((psi[:].real)**2+(psi[:].imag)**2))) +
                                            "\n")

                elif self.params['wavepacket_format'] == "h5":

                    fl_wavepacket.create_dataset(   name        = str('{:10.3f}'.format(t)), 
                                                    data        = psi,
                                                    dtype       = complex,
                                                    compression = 'gzip' #no-loss compression. Compression with loss is possible and can save space.
                                                )
            
            end_time = time.time()
            print("time per timestep =  " + str("%10.3f"%(end_time-start_time)) + "s")
    
        
        end_time_global = time.time()
        print("The time for the wavefunction propagation is: " + str("%10.3f"%(end_time_global-start_time_global)) + "s")
        fl_wavepacket.close()



def PROJECT_HAM_GLOBAL(params, maparray, Nbas, Gr, ham0, grid_euler, irun):

    ham = sparse.csr_matrix((Nbas, Nbas), dtype=complex) 
    print("size of bound hamiltonian = " + str(Nbas0))
    # 0. Append the bound hamiltonian
    #ham[:Nbas0,:Nbas0] = ham0

    #print(ham.todense())
    #plt.spy(ham,precision=1e-8, markersize=2)
    #plt.show()



    #plt.spy(ham,precision=1e-4, markersize=2)
    #plt.show()

    #bound.plot_mat(ham.todense())

    # 2. Optional: add "long-range potential" 
    # Build the full potential in propagation space minus bound spac
        # consider cut-offs for the electrostatic potential 

    if params['molec_name'] == "chiralium": 
        potmat, potind = bound.BUILD_POTMAT_ANTON_ROT( params, maparray, Nbas , Gr, grid_euler, irun )

        potind = np.asarray(potind,dtype=int)
        """ Put the indices and values back together in the Hamiltonian array"""
        for ielem, elem in enumerate(potmat):
            #print(elem[0])
            ham[ potind[ielem,0], potind[ielem,1] ] = elem[0]

    elif params['molec_name'] == "h": # test case of shifted hydrogen
        potmat, potind = bound.BUILD_POTMAT0_ROT( params, maparray, Nbas, Gr, grid_euler, irun ) 

        potind = np.asarray(potind,dtype=int)
        """ Put the indices and values back together in the Hamiltonian array"""
        for ielem, elem in enumerate(potmat):
            #print(elem[0])
            ham[ potind[ielem,0], potind[ielem,1] ] = elem[0]

    # 1. Build the full KEO in propagation space minus bound space
 
    start_time = time.time()
    keomat = bound.BUILD_KEOMAT_FAST( params, maparray, Nbas , Gr )
    end_time = time.time()
    print("Time for construction of KEO matrix in full propagation space is " +  str("%10.3f"%(end_time-start_time)) + "s")

    #plt.spy(keomat,precision=1e-8, markersize=2)
    #plt.show()

    #print("Shape of keomat: " + str(keomat.shape) )
    #print("Shape of ham: " + str(ham.shape) )

    #keomat_copy = keomat.copy()
    #keomat_copy += keomat.getH()
    #for i in range(keomat.shape[0]):
    #    keomat_copy[i,i] /=2.0 #-= hmat.diagonal()[i]


    #ham[:Nbas0,:Nbas0]  -= keomat_copy[:Nbas0, :Nbas0]
    #plt.spy(ham,precision=1e-4, markersize=2)
    #plt.show()

    ham += keomat#_copy  


    hmat_csr_size = ham.data.size/(1024**2)
    print('Size of the sparse Hamiltonian csr_matrix: '+ '%3.2f' %hmat_csr_size + ' MB')
    ham_copy = ham.copy()
    ham_copy += ham.getH()
    for i in range(ham.shape[0]):
        ham_copy[i,i] /=2.0#-= hmat.diagonal()[i]


    #apply filters

    """ --- filter hamiltonian matrix  --- """

    if params['hmat_format'] == 'numpy_arr':    
        ham_filtered = np.where( np.abs(ham_copy) < params['hmat_filter'], 0.0, ham_copy)
        #ham_filtered = sparse.csr_matrix(ham_filtered)

    elif params['hmat_format'] == 'sparse_csr':
        nonzero_mask        = np.array(np.abs(ham_copy[ham_copy.nonzero()]) < params['hmat_filter'])[0]
        rows                = ham_copy.nonzero()[0][nonzero_mask]
        cols                = ham_copy.nonzero()[1][nonzero_mask]
        ham_copy[rows, cols]    = 0
        ham_filtered        = ham_copy.copy()


    #assert TEST_boundARY_HAM(params,ham_copy,Nbas0) == True, "Oh no! The bound Hamiltonian is incompatible with the full Hamiltonian."
    

    return ham_filtered


def PROJECT_PSI_GLOBAL(params, maparray, psi0):
    Nbas = len(maparray)
    Nbas0 = len(psi0)
    print("Nbas = " + str(Nbas) + ", Nbas0 = " + str(Nbas0))

    psi = np.zeros(Nbas, dtype = complex)    
    psi[:Nbas0] = psi0[:,params['ivec']]

    return Nbas, psi

def TEST_boundARY_HAM(params,ham,Nbas0):
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


def BUILD_HMAT0_ROT(params, grid_euler, irun):
    """ Build the bound Hamiltonian with rotated ESP in unrotated basis, store the hamiltonian in a file 
    
    
    
    
    
    
    """
    
    print("Nbas0 = " + str(Nbas0))
    print("maparray0: ")
    print(maparray)
    
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

                #bound.plot_mat(hmat)
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
            potmat, potind = bound.BUILD_POTMAT0_ROT( params, maparray, Nbas, Gr, grid_euler, irun )   


        elif params['esp_mode'] == "multipoles":
            potmat, potind = bound.BUILD_POTMAT0_MULTIPOLES_ROT( params, maparray, Nbas , Gr, grid_euler, irun )
        
        elif params['esp_mode'] == "anton":
            potmat, potind = bound.BUILD_POTMAT0_ANTON_ROT( params, Nbas , Gr, grid_euler, irun )

        """ Put the indices and values back together in the Hamiltonian array"""
        for ielem, elem in enumerate(potmat):
            #print(elem[0])
            hmat[ potind[ielem][0], potind[ielem][1] ] = elem[0]


        #print("plot of hmat")

        #bound.plot_mat(hmat.todense())
        #plt.spy(hmat,precision=params['sph_quad_tol'], markersize=3, label="HMAT")
        #plt.legend()
        #plt.show()
        #exit()

        """ calculate KEO """
        start_time = time.time()
        #print(Gr.ravel())
        #exit()
        keomat = bound.BUILD_KEOMAT_FAST( params, maparray, Nbas , Gr )
        end_time = time.time()
        print("New implementation - time for construction of KEO matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")

        #start_time = time.time()
        #keomat = bound.BUILD_KEOMAT( params, maparray, Nbas , Gr )
        #end_time = time.time()
        #print("Old implementation - time for construction of KEO matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")
        
        hmat += keomat 
        #bound.plot_mat(hmat.todense())
        #print("plot of hmat")
        #bound.plot_mat(hmat)
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
                np.savetxt( energyfile, enr * constants.au_to_ev , fmt='%10.5f' )
    

        """ Plot initial orbitals """
        if params['plot_ini_orb'] == True:
            plots.plot_initial_orbitals(params,maparray,coeffs)


        return ham_filtered, coeffs

def call_eigensolver(A,params):
    if params['ARPACK_enr_guess'] == None:
        print("No eigenvalue guess defined")
    else:
        params['ARPACK_enr_guess'] /= constants.au_to_ev


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
        coeffs.append([i,n,xi,l,m,np.asarray(c,dtype=complex)])
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


def calc_intmat(maparray,rgrid,Nbas, helicity):  

    start_time = time.time()

    #field: (E_-1, E_0, E_1) in spherical tensor form
    """calculate the <Y_l1m1(theta,phi)| d(theta,phi) | Y_l2m2(theta,phi)> integral """
    #Adopted convention used in Artemyev et al., J. Chem. Phys. 142, 244105 (2015).
    
    if params['hmat_format'] == 'numpy_arr':    
        intmat =   np.zeros(( Nbas , Nbas ), dtype = float)
    elif params['hmat_format'] == 'sparse_csr':
        intmat = sparse.csr_matrix(( Nbas, Nbas ), dtype = float)
        #intmathc = sparse.csr_matrix(( Nbas, Nbas ), dtype = float) #for symmetry testing

    #D = np.zeros(3, dtype = float)

    """precompute all necessary 3-j symbols"""
    #generate arrays of 3j symbols with 'spherical':
    start_time = time.time()
    tjmat_CG = gen_3j_dip(params['bound_lmax'], "CG")
    end_time = time.time()
    print("time for calculation of tjmat in dipole matrix =  " + str("%10.3f"%(end_time-start_time)) + "s")
    
    #start_time = time.time()
    #tjmat_tj = gen_3j_dip(params['bound_lmax'], "tj")
    #end_time = time.time()
    #print("time for calculation of tjmat in dipole matrix =  " + str("%10.3f"%(end_time-start_time)) + "s")

    #determine sigma
    #if helicity == "R":
    #    sigma = 1
    #elif helicity == "L":
    #    sigma = -1 
    #elif helicity == "0":
    #   sigma = 0
    #else:
    #    ValueError("incorrect helicity")

    #Generate global diplistt
    start_time = time.time()
    """ Note: we always (for R/L CPL) produce diplist for sigma = +1 and generate elements with sigma = -1 with the hermitian conjugate"""
    diplist = mapping.GEN_DIPLIST_opt1(maparray, Nbas, params['bound_lmax'], params['map_type'] ) # new, non-vectorized O(n) implementation
    #diplist = mapping.GEN_DIPLIST( maparray, Nbas, params['map_type'], sigma ) #old O(n**2) implementation
    diplist = np.asarray(diplist,dtype=int)
    end_time = time.time()
    print("Time for the construction of diplist: " +  str("%10.3f"%(end_time-start_time)) + "s")
    #print(diplist)
    #exit()

    #rtol=1e-05
    #atol=1e-08
    #isclose = np.allclose(tjmat_CG, tjmat_tj, rtol=rtol, atol=atol)
    #print("Are tjmat_CG and tjmat_tj identical? " + str(isclose))

    rgrid = rgrid.ravel()

    for i in range(diplist.shape[0]):
        #this step can be made more efficient by vectorisation based on copying the tjmat block and overlaying rgrid
        intmat[ diplist[i,5], diplist[i,6] ] =  rgrid[ diplist[i][0] - 1 ] * tjmat_CG[ diplist[i,1], diplist[i,3], diplist[i,1]+diplist[i,2], diplist[i,3]+diplist[i,4], 2] #sigma+1
        
        #intmathc[ diplist[i,5], diplist[i,6] ] = rgrid[ diplist[i][0] - 1 ] * tjmat_CG[ diplist[i,3], diplist[i,1], diplist[i,3]+diplist[i,4], diplist[i,1]+diplist[i,2], 2 ]


    #plt.spy(intmathc-intmat, precision=1e-12, markersize=7, color='r')
    #plt.spy(intmat, precision=params['sph_quad_tol'], markersize=5, color='b')

    #plt.show()
    #   exit()

    #isclose = np.allclose(intmat.todense(),intmathc.todense())
    #print("Is tjmat[sigma].hc identical to (-1) * tjmat[-sigma]? " + str(isclose))
    #exit()
    """
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

                D[:] = tjmat_tj[ maparray[i][3], maparray[j][3], maparray[i][4] + maparray[i][3], maparray[j][4] + maparray[j][3], : ] #-1, 0 , +1

                if maparray[j][4] == maparray[i][4]:
                    intmat2[i,j] = np.sqrt( 2.0 * np.pi / 3.0 ) * D[1] * rin * np.sqrt(2.)

                elif maparray[j][4] ==  maparray[i][4] + 1: 
                    intmat1[i,j] = np.sqrt( 2.0 * np.pi / 3.0 ) * D[0] * rin 

                elif maparray[j][4] == maparray[i][4] - 1:
                    intmat3[i,j] = np.sqrt( 2.0 * np.pi / 3.0 ) * D[2] * rin 

    
    #plt.spy(intmat_new1, precision=params['sph_quad_tol'], markersize=5)
    plt.spy(intmat1, precision=params['sph_quad_tol'], markersize=5, color='r')
    plt.spy(intmat2, precision=params['sph_quad_tol'], markersize=5, color='g')
    plt.spy(intmat3, precision=params['sph_quad_tol'], markersize=5, color='b')
    plt.show()
    exit()
    #rtol=1e-05
    #atol=1e-08
    #print(intmat_new)

    #np.allclose(intmat, intmat_new, rtol=rtol, atol=atol)


    #intmat += np.conjugate(intmat.T)

    #print("Interaction matrix")
    #with np.printoptions(precision=4, suppress=True, formatter={'complex': '{:15.8f}'.format}, linewidth=400):
    #    print(intmat)
    #print("Is the interaction matrix symmetric? " + str(check_symmetric(intmat)))
    """
    #see derivation for the "-" sign in front of intmat


    intmat_hc = intmat.copy()
    intmat_hc = intmat_hc.transpose() #d_ll'mm',mu=-1

    #determine sigma
    if helicity == "R" or helicity =="L":
        sigma = 2
    elif helicity == "0":
        sigma = 1
        intmat /= 2.0
    else:
        ValueError("incorrect helicity")
    
    end_time = time.time()
    print("time for calculation of dipole interaction matrix =  " + str("%10.3f"%(end_time-start_time)) + "s")


    return (-1.0) * np.sqrt( 2.0 * np.pi / 3.0 ) * intmat #1,intmat2,intmat3


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.H, rtol=rtol, atol=atol)

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


def gen_3j_dip(lmax,mode):
    """precompute all necessary 3-j symbols for dipole matrix elements"""

    # 2) tjmat[l1,l2,m1,m2,sigma] = [0,...lmax,0...lmax,0,...,m+l,0...2]
    tjmat = np.zeros( (lmax + 1, lmax + 1, 2*lmax + 1, 2*lmax + 1, 3), dtype = float)
    #Both modes checked with textbooks and verified that produce identical matrix elements. 15 Dec 2021.

    if mode == "CG":

        for mu in range(0,3):
            for l1 in range(lmax+1):
                for l2 in range(lmax+1):
                    for m1 in range(-l1,l1+1):
                        for m2 in range(-l2,l2+1):

                            tjmat[l1,l2,l1+m1,l2+m2,mu] = np.sqrt( (2.0*float(l2)+1) * (3.0) /(  (2.0*float(l1)+1) * (4.0*np.pi) ) ) * spherical.clebsch_gordan(l2,m2,1,mu-1,l1,m1) * spherical.clebsch_gordan(l2,0,1,0,l1,0)
                        #tjmat[l1,l2,l1+m1,l2+m2,mu] = spherical.Wigner3j(l1, 1, l2, m, mu-1, -(m+mu-1)) * spherical.Wigner3j(l1, 1, l2, 0, 0, 0)
                        #tjmat[l1,l2,l1+m,mu] *= np.sqrt((2*float(l1)+1) * (2.0*float(1.0)+1) * (2*float(l2)+1)/(4.0*np.pi)) * (-1)**(m+mu-1)

    elif mode == "tj":
        for mu in range(0,3):
            for l1 in range(lmax+1):
                for l2 in range(lmax+1):
                    for m1 in range(-l1,l1+1):
                        for m2 in range(-l2,l2+1):

                            tjmat[l1,l2,l1+m1,l2+m2,mu] = spherical.Wigner3j(l2, 1, l1, m2, mu-1, -m1) * spherical.Wigner3j(l2, 1, l1, 0, 0, 0) *\
                                                            np.sqrt((2*float(l1)+1) * (2.0*float(1.0)+1) * (2*float(l2)+1)/(4.0*np.pi)) * (-1)**(m1)



    #print("3j symbols in array:")
    #print(tjmat)
    return tjmat


if __name__ == "__main__":   


    
    start_time_total = time.time()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    print("\n")
    print("----------------------------------------------------------- ")
    print("---------------------- START PROPAGATE --------------------")
    print("----------------------------------------------------------- ")
    print("\n")
    
    ibatch  = int(sys.argv[1]) # id of the batch in the Euler's angles grid run
    os.chdir(sys.argv[2])
    path    = os.getcwd()

    print("current working directory: " + path)

    with open('input_prop', 'r') as input_file:
        params = json.load(input_file)

    print("\n")
    print("----------------------------------------------------------- ")
    print("------------------------ INPUT ECHO -----------------------")
    print("----------------------------------------------------------- ")
    print("\n")

    for key, value in params.items():
        print(key, ":", value)

    #PropObj = Propagator(params,0)
    #tgrid,dt = PropObj.gen_timegrid()
    
    #print(len(tgrid))
    #print(tgrid/constants.time_to_au[ params['time_units'] ])
    #print(tgrid[1]-tgrid[0])
    #exit()

    print("\n")
    print("Generating index maps...")
    print("\n")

    MapObjBound = wavefunction.Map( params['FEMLIST_bound'],
                                    params['bound_lmax'],
                                    params['map_type'],
                                    params['job_directory'])

    MapObjBoundRad = wavefunction.Map(  params['FEMLIST_bound'],
                                        0,
                                        params['map_type'],
                                        params['job_directory'])

    MapObjProp = wavefunction.Map(  params['FEMLIST_PROP'],
                                    params['bound_lmax'],
                                    params['map_type'],
                                    params['job_directory'] )

    # Note: it is more convenient to store maps and grids in params dictionary than to set up as global variables generated in a separate module/function.
    #       Now we need to generate them only once, otherwise we would need to call the generating function in each separate module.
    params['maparray0'], params['Nbas0'] = MapObjBound.GENMAP_FEMLIST()

    params['maparray'], params['Nbas']  = MapObjProp.GENMAP_FEMLIST()

    params['maparray0_rad'], params['Nbas0_rad'] = MapObjBoundRad.GENMAP_FEMLIST()
                                    
    MapObjBound.save_map(params['maparray0'], 'map_bound.dat')
    MapObjBoundRad.save_map(params['maparray0_rad'],  'map_bound_rad.dat')    
    MapObjProp.save_map(params['maparray'], 'map_prop.dat')


    print("\n")
    print("Generating grids...")
    print("\n")

    GridObjBound = wavefunction.Grid(   params['bound_nlobs'], 
                                        params['bound_nbins'] , 
                                        params['bound_binw'],  
                                        params['bound_rshift']) 

    GridObjProp = wavefunction.Grid(    params['bound_nlobs'], 
                                        params['prop_nbins'] , 
                                        params['bound_binw'],  
                                        params['bound_rshift'])

    params['Gr0'], params['Nr0']        = GridObjBound.gen_grid()
    params['Gr'], params['Nr']          = GridObjProp.gen_grid()


    print("\n")
    print("Reading grid of molecular orientations (Euler angles)...")
    print("\n")

    GridObjEuler = wavefunction.GridEuler(  params['N_euler'],
                                            params['N_batches'],
                                            params['orient_grid_type'])
    
    grid_euler, N_Euler, N_per_batch  = GridObjEuler.read_euler_grid()


    print("\n")
    print("Setting up Hamiltonians...")
    print("\n")

    HamObj = hamiltonian.Hamiltonian(params)

    print("\n")
    print("Building the kinetic energy operator matrix for the bound Hamiltonian...")
    print("\n")

    KEO_bound = HamObj.build_keo_bound()


    print("\n")
    print("Building the kinetic energy operator matrix for the propagation Hamiltonian...")
    print("\n")

    KEO_prop = HamObj.build_keo_prop()

    print("\n")
    print("Building the interaction matrix...")
    print("\n")

    intmat =  HamObj.build_intmat() 


    for irun in range(ibatch * N_per_batch, (ibatch+1) * N_per_batch):

        #print(grid_euler[irun])

        ham0, psi0 = HamObj.BUILD_HMAT0_ROT(params, grid_euler, irun)

        Nbas, psi_init  = PsiObj.PROJECT_PSI_GLOBAL(params,psi0) 
        ham_init        = HamObjPROJECT_HAM_GLOBAL(params, ham0 ,grid_euler, irun)

        PropObj = Propagator(params,irun)

        PropObj.prop_wf(params, ham_init, psi_init, irun)


    end_time_total = time.time()
    print("Global time =  " + str("%10.3f"%(end_time_total-start_time_total)) + "s")
