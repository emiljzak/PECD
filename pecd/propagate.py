#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2022 Emil Zak <emil.zak@cfel.de>, <emil.j.zak@gmail.com>
#
from textwrap import indent
from tkinter import Grid
from h5py._hl import datatype
import numpy as np
from scipy import sparse
from scipy.fftpack import fftn
from scipy.sparse.linalg import expm_multiply,expm
from scipy.sparse.linalg import eigsh
from scipy.special import sph_harm
from scipy.special import eval_legendre
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


#special acceleration libraries
from numba import njit, prange

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




    def save_wavepacket_attrs(self,fl_wavepacket):

        wfn = fl_wavepacket.create_dataset( name        = "metadata", 
                                            data        = np.zeros(1),
                                            dtype       = int)

        wfn.attrs['t0']         = self.params['t0']
        wfn.attrs['tmax']       = self.params['tmax']
        wfn.attrs['dt']         = self.params['dt']
        wfn.attrs['saverate']   = self.params['wfn_saverate']
        wfn.attrs['units']      = self.params['time_units']


    def save_wavepacket(self,fl_wavepacket,itime,psi):

        if self.params['wavepacket_format'] == "dat":
                        
            fl_wavepacket.write(    '{:10d}'.format(itime) + 
                                    " ".join('{:15.5e}'.format(psi[i].real) + 
                                    '{:15.5e}'.format(psi[i].imag) for i in range(0,self.Nbas)) +
                                    '{:15.8f}'.format(np.sqrt(np.sum((psi[:].real)**2+(psi[:].imag)**2))) +
                                    "\n")

        elif self.params['wavepacket_format'] == "h5":

            fl_wavepacket.create_dataset(   name        = str(itime), 
                                            data        = psi,
                                            dtype       = complex,
                                            compression = 'gzip' #no-loss compression. Compression with loss is possible and can save space.
                                        )


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
 
    @staticmethod
    def expv_lanczos(vec, t, matvec, maxorder=1000, tol=1e-12):
        """ Computes epx(t*a)*v using Lanczos with cupy mat-vec product """

        V, W = [], []
        T = np.zeros((maxorder, maxorder), dtype=vec.dtype)

        # first Krylov basis vector
        V.append(vec)
        w = matvec(V[0])
        T[0, 0] = np.vdot(w, V[0])
        W.append(w - T[0, 0] * V[0])

        # higher orders
        u_kminus1, u_k, conv_k, k = {}, V[0], 1, 1
        while k < maxorder and conv_k > tol:

            # extend ONB of Krylov subspace by another vector
            T[k - 1, k] = np.sqrt(np.sum(np.abs(W[k - 1])**2))
            T[k, k - 1] = T[k - 1, k]
            if not T[k - 1, k] == 0:
                V.append(W[k - 1] / T[k - 1, k])

            # reorthonormalize ONB of Krylov subspace, if neccesary
            else:
                v = np.ones(V[k - 1].shape, dtype=np.complex128)
                for j in range(k):
                    proj_j = np.vdot(V[j], v)
                    v = v - proj_j * V[j]
                norm_v = np.sqrt(np.sum(np.abs(v)**2))
                V.append(v / norm_v)

            w = matvec(V[k])
            T[k, k] = np.vdot(w, V[k])
            w = w - T[k, k] * V[k] - T[k - 1, k] * V[k - 1]
            W.append(w)

            # calculate current approximation and convergence
            u_kminus1 = u_k
            expT_k = expm(t * T[: k + 1, : k + 1])
            u_k = sum([expT_k[i, 0] * v_i for i,v_i in enumerate(V)])
            conv_k = np.sum(np.abs(u_k - u_kminus1)**2)

            k += 1

        if k == maxorder:
            print("lanczos reached maximum order of {}".format(maxorder))

        return u_k


    @staticmethod
    def matvec_numpy(matrix, counter):
        """ Computes lambda function for mat-vec prodcut with numba """
        # define mat-vec product with numpy
        def matvec(v, counter_):
            u = matrix.dot(v)
            counter_ += 1
            return u

        # create lambda function of above mat-vec product
        matvec_ = lambda v : matvec(v, counter)

        return matvec_

    @staticmethod
    def matvec_numba(matrix):
        """ Computes lambda function for mat-vec prodcut with numba """
        # get components of sparse matrix
        data = matrix.data
        indices = matrix.indices
        indptr = matrix.indptr

        # define mat-vec product with numba
        @njit(parallel=True)
        def matvec(v):
            u = np.zeros(v.size, dtype=v.dtype)
            for i in prange(v.size):
                for j in prange(indptr[i], indptr[i + 1]):
                    u[i] += data[j] * v[indices[j]]
            return u

        # create lambda function of above mat-vec product
        matvec_ = lambda v : matvec(v)

        return matvec_



    @staticmethod
    def expv_taylor(vec, t, matvec, maxorder=1000, tol=0):
        """ Computes epx(t*a)*v using Taylor with cupy mat-vec product """

        V = []

        # zeroth order
        V.append(vec)

        # higher orders
        conv_k, k = 1, 0
        while k < maxorder and conv_k > tol:
            k += 1
            v = matvec(V[k - 1]) * t / k / 2**k
            conv_k = np.sum(np.abs(v)**2)
            V.append(v)

        if k == maxorder:
            print("taylor reached maximum order of {}".format(maxorder))

        u = sum(V)

        return u

    def prop_wf(self, ham_init, dipmat, psi_init):
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

        helicity = hamiltonian.Hamiltonian.pull_helicity(self.params)
        fl_wavepacket = self.allocate_wavepacket(helicity)

        print("\n")
        print("Saving wavepacket attributes...")
        print("\n")

        self.save_wavepacket_attrs(fl_wavepacket)

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

        counter = np.zeros(1)

        start_time_global = time.time()
        for itime, t in enumerate(tgrid): 

            start_time = time.time()

            if itime%10 == 0:
                print("t = " + str( "%10.1f"%(t/self.time_to_au)) + " as" + " normalization: " + str(np.sqrt( np.sum( np.conj(psi) * psi )) ) ) 
            else:
                print("t = " + str( "%10.1f"%(t/self.time_to_au)) + " as")
        
            # building the electric dipole interaction matrix
            dip = Fvec[0,itime] * dipmat - Fvec[2,itime] * dipmat.H #+ Fvec[1,itime] * dipmat #for LP
           
            #dip = sparse.csr_matrix(dip)
            #print("Is the full hamiltonian matrix symmetric? " + str(check_symmetric(ham_init.todense() + dip.todense() )))
            #exit()
            
            #Note: we can use multiple time-points with expm_multiply and ham_init as linear operator. Also action on a collection of vectors is possible.
            
            #psi = quimb.linalg.base_linalg.expm_multiply(-1.0j * ( ham_init + dip ) * dt, psi, backend='SCIPY')
            #matvec_nu = Propagator.matvec_numba(ham_init + dip)
            matvec_np = Propagator.matvec_numpy( ( ham_init + dip ), counter)
            #psi_new = Propagator.expv_taylor(psi, dt, matvec_np,tol=1e-10)
            psi = Propagator.expv_lanczos(psi, -1.0j *dt, matvec_np, tol=1e-10)
            #psi = psi_new
            #psi = expm_multiply(-1.0j * ( ham_init + dip ) * dt , psi ) 

            if itime%self.wfn_saverate == 0: self.save_wavepacket(fl_wavepacket,itime,psi)

            end_time = time.time()
            print("time per timestep =  " + str("%10.3f"%(end_time-start_time)) + "s")
    
        
        end_time_global = time.time()
        print("The time for the wavefunction propagation is: " + str("%10.3f"%(end_time_global-start_time_global)) + "s")
        fl_wavepacket.close()





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

    MapObjPropRad = wavefunction.Map(  params['FEMLIST_PROP'],
                                        params['map_type'],
                                        params['job_directory'])

    # Note: it is more convenient to store maps and grids in params dictionary than to set up as global variables generated in a separate module/function.
    #       Now we need to generate them only once, otherwise we would need to call the generating function in each separate module.
    params['maparray0'], params['Nbas0']            = MapObjBound.genmap_femlist()
    params['maparray'], params['Nbas']              = MapObjProp.genmap_femlist()
    params['maparray_rad'], params['Nbas_rad']      = MapObjPropRad.genmap_femlist()
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
    print("Building Wigner-D matrices for rotations...")
    print("\n")
    WDMATS_wfn = GridObjEuler.gen_wigner_dmats(N_Euler, grid_euler,params['bound_lmax'])
    WDMATS_pot = GridObjEuler.gen_wigner_dmats(N_Euler, grid_euler,params['multi_lmax'])
    
    print("\n")
    print("Setting up Hamiltonians...")
    print("\n")

    HamObjBound = hamiltonian.Hamiltonian(params,WDMATS_pot,'bound')
    HamObjProp  = hamiltonian.Hamiltonian(params,WDMATS_pot,'prop')

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
    enr_bound_0, psi_bound_0 = HamObjBound.call_eigensolver(ham_bound)
    end_time = time.time()
    print("Time for the calculation of eigenvalues and eigenvectos of the bound Hamiltonian = " +  str("%10.3f"%(end_time-start_time)) + "s")
    
    
    print("Saving energy levels and wavefunctions for the bound Hamiltonian...")
    if params['save_enr0'] == True: HamObjBound.save_energies(enr_bound_0)
    if params['save_psi0'] == True: HamObjBound.save_wavefunctions(psi_bound_0)

    print("\n")
    print("Building the interaction matrix...")
    print("\n")

    start_time = time.time()
    dipmat =  HamObjProp.build_intmat() 
    end_time = time.time()
    print("Time for the calculation of the electric dipole interaction matrix in propagation space = " +  str("%10.3f"%(end_time-start_time)) + "s")
    

    print("\n")
    print("Building the kinetic energy operator matrix for the propagation Hamiltonian...")
    print("\n")

    start_time = time.time()
    keo_prop = HamObjProp.build_keo()
    end_time = time.time()
    print("Time for the calculation of the kinetic energy matrix for the propagatino Hamiltonian = " +  str("%10.3f"%(end_time-start_time)) + "s")

    if params['restart'] == False:
        # loop over molecular orientations in the present batch
        for irun in range(ibatch * N_per_batch, (ibatch+1) * N_per_batch):

            start_time  = time.time()
            pot_prop    = HamObjProp.build_potmat(grid_euler, irun)
            end_time    = time.time()
            print("Time for construction of the potential energy matrix for the propagation Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")

            start_time  = time.time()
            ham_init    = HamObjProp.build_ham(keo_prop, pot_prop)
            end_time    = time.time()
            print("Time for filtering of the propagation Hamiltonian at time t=0: " +  str("%10.3f"%(end_time-start_time)) + "s")

            PsiObj          = wavefunction.Psi(params,WDMATS_wfn,grid_euler,irun,"bound")
            psi_bound_0_rot = PsiObj.rotate_psi(psi_bound_0[:,params['ivec']])
            psi_init        = PsiObj.project_psi_global(psi_bound_0_rot)

            PropObj     = Propagator(params,irun)
            PropObj.prop_wf(ham_init, dipmat, psi_init)

    else:
        #read restart list
        for irun in params['restart_list']:
            print(type(irun))
            print(irun)
            start_time  = time.time()
            pot_prop    = HamObjProp.build_potmat(grid_euler, irun)
            end_time    = time.time()
            print("Time for construction of the potential energy matrix for the propagation Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")

            start_time  = time.time()
            ham_init    = HamObjProp.build_ham(keo_prop, pot_prop)
            end_time    = time.time()
            print("Time for filtering of the propagation Hamiltonian at time t=0: " +  str("%10.3f"%(end_time-start_time)) + "s")

            PsiObj          = wavefunction.Psi(params,WDMATS_wfn,grid_euler,irun,"bound")
            psi_bound_0_rot = PsiObj.rotate_psi(psi_bound_0[:,params['ivec']])
            psi_init        = PsiObj.project_psi_global(psi_bound_0_rot)

            PropObj     = Propagator(params,irun)
            PropObj.prop_wf(ham_init, dipmat, psi_init)



    end_time_total = time.time()
    print("Global time =  " + str("%10.3f"%(end_time_total-start_time_total)) + "s")