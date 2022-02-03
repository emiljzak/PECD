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
import quimb
from sympy.core.numbers import Integer

from sympy.physics.wigner import gaunt
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.quantum.cg import CG
from sympy import N

import itertools
import json
import h5py


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
        """Pulls helicity of the laser pulse.

            Returns: str
                helicity = "R","L" or "0" for right-,left- circularly polarized light and linearly polarized light, respectively.
        """
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


    def gen_timegrid(self):

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
        norm = np.sqrt( np.sum( np.conjugate(psi[:]) * psi[:] ) )
        return psi/norm
 

    def prop_wf(self, ham_init, intmat, psi_init):
        """This function propagates the wavefunction with the time-dependent Hamiltonian and saves the wavepacket in a file.

            Here the time-grid and the electric fields are constructed. The wavefunction is propagated by an iterative
            calculation of the product of the time-dependent Hamiltonian matrix exponential and the wavefunction vector:

            .. math::
                \psi(t_{i+1}) = \exp(-i\cdot H(t) \cdot dt) \psi(t_i)
                :label: wf_propagation
            
            Arguments: tuple
                ham_init : scipy.sparse.csr_matrix, dtype = complex/float, shape = (Nbas,Nbas)
                    Initial Hamiltonian matrix

                intmat : scipy.sparse.csr_matrix, dtype = complex, shape = (Nbas,Nbas)
                    electric dipole interaction matrix (with no field included)
                
                psi_init : numpy.ndarray, dtype = complex, shape = (Nbas,)
                    initial wavefunction

            Returns: None

            .. note:: The initial Hamiltonian `ham_init` :py:func:`Hamiltonian.build_ham` has complex matrix elements in general. However the kinetic energy matrix constructed in :py:func:`Hamiltonian.build_keo` is real by construction. Therefore when the potential matrix calculated in :py:func:`Hamiltonian.build_potmat` is real, the Hamiltonian matrix is real too.

        """
        #self.calc_mat_density(ham_init)
        ##self.spy_mat(ham_init)
        #self.plot_mat(ham_init)

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
        Fvec     = FieldObj.gen_field(tgrid) 
        
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
            dip = Fvec[0,itime] * intmat - Fvec[2,itime] * intmat.H #+ Fvec[itime,1] * intmat
           
            #dip = sparse.csr_matrix(dip)
            #print("Is the full hamiltonian matrix symmetric? " + str(check_symmetric(ham_init.todense() + dip.todense() )))
            #exit()
            
            #Note: we can use multiple time-points with expm_multiply and ham_init as linear operator. Also action on a collection of vectors is possible.
            
            psi = quimb.linalg.base_linalg.expm_multiply(-1.0j * ( ham_init + dip ) * dt, psi, backend='SCIPY')
            #psi = expm_multiply( -1.0j * ( ham_init + dip ) * dt, psi ) 


            if itime%self.wfn_saverate == 0:

                if self.params['wavepacket_format'] == "dat":
                    
                    fl_wavepacket.write(    '{:10.3f}'.format(t) + 
                                            " ".join('{:15.5e}'.format(psi[i].real) + 
                                            '{:15.5e}'.format(psi[i].imag) for i in range(0,self.Nbas)) +
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

    print("\n")
    print("Generating index maps...")
    print("\n")

    MapObjBound = wavefunction.Map( params['FEMLIST_BOUND'],
                                    params['map_type'],
                                    params['job_directory'],
                                    params['bound_lmax'])

    MapObjBoundRad = wavefunction.Map(  params['FEMLIST_BOUND'],
                                        params['map_type'],
                                        params['job_directory'])

    MapObjProp = wavefunction.Map(  params['FEMLIST_PROP'],
                                    params['map_type'],
                                    params['job_directory'],
                                    params['bound_lmax'] )

    # Note: it is more convenient to store maps and grids in params dictionary than to set up as global variables generated in a separate module/function.
    #       Now we need to generate them only once, otherwise we would need to call the generating function in each separate module.
    params['maparray0'], params['Nbas0']            = MapObjBound.genmap_femlist()
    params['maparray'], params['Nbas']              = MapObjProp.genmap_femlist()
    params['maparray0_rad'], params['Nbas0_rad']    = MapObjBoundRad.genmap_femlist()
                                    
    MapObjBound.save_map(params['maparray0'], 'map_bound.dat')
    MapObjBoundRad.save_map(params['maparray0_rad'],  'map_bound_rad.dat')    
    MapObjProp.save_map(params['maparray'], 'map_prop.dat')


    print("\n")
    print("Generating grids...")
    print("\n")

    GridObjBound = wavefunction.GridRad(    params['bound_nlobs'], 
                                            params['bound_nbins'] , 
                                            params['bound_binw'],  
                                            params['bound_rshift']) 

    GridObjProp = wavefunction.GridRad(     params['bound_nlobs'], 
                                            params['prop_nbins'] , 
                                            params['bound_binw'],  
                                            params['bound_rshift'])

    params['x'], params['w'], params['rgrid0_prim'], params['rgrid0'], params['Nr0']    = GridObjBound.gen_grid()
    _, _, params['rgrid_prim'], params['rgrid'], params['Nr']                           = GridObjProp.gen_grid()
    

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

    HamObjBound = hamiltonian.Hamiltonian(params,'bound')
    HamObjProp  = hamiltonian.Hamiltonian(params,'prop')

    print("\n")
    print("Building the kinetic energy operator matrix for the bound Hamiltonian...")
    print("\n")

    start_time = time.time()
    keo_bound = HamObjBound.build_keo()
    end_time = time.time()
    print("Time for construction of the kinetic energy matrix for bound Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")
  

    print("\n")
    print("Building the potential energy operator matrix for the bound Hamiltonian...")
    print("\n")

    start_time = time.time()
    pot_bound = HamObjBound.build_potmat()
    end_time = time.time()
    print("Time for construction of the potential energy matrix for bound Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")
    
    
    print("\n")
    print("Building the bound Hamiltonian operator matrix...")
    print("\n")

    start_time = time.time()
    ham_bound = HamObjBound.build_ham(keo_bound,pot_bound)
    end_time = time.time()
    print("Time for construction of the bound Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")


    print("\n")
    print("Calculating eigenvalues and eigenvectors of the bound Hamiltonian operator matrix...")
    print("\n")

    start_time = time.time()
    E0, psi0 = call_eigensolver(ham_bound,params)
    end_time = time.time()
    print("Time for the calculation of eigenvalues and eigenvectos of the bound Hamiltonian = " +  str("%10.3f"%(end_time-start_time)) + "s")
    
    
    print("Energy levels:")
    print(E0*constants.au_to_ev )
    #exit()

    print("\n")
    print("Building the interaction matrix...")
    print("\n")


    start_time = time.time()
    intmat =  HamObjProp.build_intmat() 
    end_time = time.time()
    print("Time for the calculation of the electric dipole interaction matrix in propagation space = " +  str("%10.3f"%(end_time-start_time)) + "s")
    

    print("\n")
    print("Building the kinetic energy operator matrix for the propagation Hamiltonian...")
    print("\n")

    start_time = time.time()
    keo_prop = HamObjProp.build_keo()
    end_time = time.time()
    print("Time for the calculation of the kinetic energy matrix for the propagatino Hamiltonian = " +  str("%10.3f"%(end_time-start_time)) + "s")
    

    # loop over molecular orientations in the present batch
    for irun in range(ibatch * N_per_batch, (ibatch+1) * N_per_batch):

        start_time  = time.time()
        pot_prop    = HamObjProp.build_potmat(grid_euler, irun)
        end_time    = time.time()
        print("Time for construction of the potential energy matrix for the propagation Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")

        start_time  = time.time()
        ham_init    = HamObjProp.build_ham(keo_prop,pot_prop)
        end_time    = time.time()
        print("Time for filtering of the propagation Hamiltonian at time t=0: " +  str("%10.3f"%(end_time-start_time)) + "s")
   
        print("\n")
        print("Setting up initial wavefunction for irun = " + str(irun) + " with Euler angles: " + str(grid_euler[irun]))
        print("\n")

        PsiObj      = wavefunction.Psi(params,grid_euler,irun)
        psi0_rot    = PsiObj.rotate_psi(psi0[:,params['ivec']])
        psi_init    = PsiObj.project_psi_global(psi0_rot)

        print("\n")
        print("Setting up initial wavefunction...")
        print("\n")
        PropObj     = Propagator(params,irun)
        PropObj.prop_wf(ham_init, intmat, psi_init)


    end_time_total = time.time()
    print("Global time =  " + str("%10.3f"%(end_time_total-start_time_total)) + "s")
