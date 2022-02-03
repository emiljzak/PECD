#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2022 Emil Zak <emil.zak@cfel.de>
#

from numpy.core.numeric import NaN
from scipy import special
from scipy.sparse.linalg import eigsh
from sympy.multipledispatch.dispatcher import RaiseNotImplementedError

import potential
import constants

import numpy as np
from numpy.linalg import multi_dot
from scipy import sparse
from scipy.special import sph_harm
import scipy.integrate as integrate
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
import quadpy

from sympy.functions.elementary.miscellaneous import sqrt

import quaternionic
import spherical

import sys
import os.path
import time

from numba import jit, prange

import matplotlib.pyplot as plt
from matplotlib import cm, colors

""" start of @jit section """
jitcache = False

@jit(nopython=True, parallel=False, cache = jitcache, fastmath=False) 
def P(l, m, x):
	pmm = np.ones(1,)
	if m>0:
		somx2 = np.sqrt((1.0-x)*(1.0+x))
		fact = 1.0
		for i in range(1,m+1):
			pmm = pmm * (-1.0 * fact) * somx2
			fact = fact + 2.0
	
	if l==m :
		return pmm * np.ones(x.shape,)
	
	pmmp1 = x * (2.0*m+1.0) * pmm
	
	if l==m+1:
		return pmmp1
	
	pll = np.zeros(x.shape)
	for ll in range(m+2, l+1):
		pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m)
		pmm = pmmp1
		pmmp1 = pll
	
	return pll

LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')
    
@jit(nopython=True, parallel=False, cache = jitcache, fastmath=False) 
def fast_factorial(n):
    return LOOKUP_TABLE[n]

def factorial(x):
	if(x == 0):
		return 1.0
	return x * factorial(x-1)

@jit(nopython=True, parallel=False, cache = jitcache, fastmath=False) 
def K(l, m):
	return np.sqrt( ((2 * l + 1) * fast_factorial(l-m)) / (4*np.pi*fast_factorial(l+m)) )

@jit(nopython=True, parallel=False, cache = jitcache, fastmath=False) 
def SH(l, m, theta, phi):
	if m==0 :
		return K(l,m)*P(l,m,np.cos(theta))*np.ones(phi.shape,)
	elif m>0 :
		return np.sqrt(2.0)*K(l,m)*np.cos(m*phi)*P(l,m,np.cos(theta))
	else:
		return np.sqrt(2.0)*K(l,-m)*np.sin(-m*phi)*P(l,-m,np.cos(theta))

#@jit( nopython=True, parallel=False, cache = jitcache, fastmath=False) 
def calc_potmat_jit( vlist, VG, Gs ):
    pot = []
    potind = []

    for p1 in range(vlist.shape[0]):
        #print(vlist[p1,:])
        w = Gs[vlist[p1,0]-1][:,2]
        G = Gs[vlist[p1,0]-1] 
        V = VG[vlist[p1,0]-1] #xi starts from 1,2,3 but python listst start from 0.

        #sph_harm( vlist[p1,2] , vlist[p1,1] , G[:,1] + np.pi,  G[:,0])
        f = np.conj(sph_harm( vlist[p1,2] , vlist[p1,1] , G[:,1] + np.pi,  G[:,0])) * \
            sph_harm( vlist[p1,4] , vlist[p1,3] , G[:,1] + np.pi,  G[:,0]) *\
            V[:]

        pot.append( [np.dot(w,f.T) * 4.0 * np.pi ] )
        potind.append( [ vlist[p1,5], vlist[p1,6] ] )

        #potmat[vlist[p1,5],vlist[p1,6]] = np.dot(w,f.T) * 4.0 * np.pi
    return pot, potind

#@jit( nopython=True, parallel=False, cache = jitcache, fastmath=False) 
def calc_potmat_multipoles_jit( vlist, tjmat, qlm, Lmax, rlmat ):
    pot = []
    potind = []

    for p in range(vlist.shape[0]):
        #print(vlist[p1,:])
        v = 0.0+1j*0.0
        for L in range(Lmax):
            for M in range(-L,L+1):
                v += qlm[(L,M)] * tjmat[vlist[p,1], L, vlist[p,3], vlist[p,2], M+L] * rlmat[vlist[p,0]-1,L]
        pot.append( [v] )
        potind.append( [ vlist[p,5], vlist[p,6] ] )

    return pot, potind




@jit( nopython=True, parallel=False, cache = jitcache, fastmath=False) 
def calc_potmatelem_xi( V, Gs, l1, m1, l2, m2 ):
    #calculate matrix elements from sphlist at point xi with V and Gs calculated at a given quadrature grid.
    w = Gs[:,2]
    #f = SH( l1 , m1  , Gs[:,0], Gs[:,1] + np.pi ) * \
    #    SH( l2 , m2  , Gs[:,0], Gs[:,1] + np.pi ) * V[:]

    f = V[:]
    return np.dot(w,f.T) * 4.0 * np.pi 









""" end of @jit section """





class Hamiltonian():
    """Hamiltonian class keeps methods for generating and manipulating matrices: kinetic energy operator, potential energy operator, hamiltonian, dipole interaction operator.

    Note:
        The characteristics of the photo-electron propagation using the FEM-DVR method suggests
        that a DVR-type mapping is suitable and natural.

    Attributes:
        hamtype (str): 'bound' or 'prop' determines the type of the hamiltonian to be created
                        

        params (dics): standard input parameters dictionary

    """

    def __init__(self,params,hamtype = 'bound'):

        self.params         = params
        self.hamtype        = hamtype

        if hamtype == "bound":
            self.Nbas       = params['Nbas0']
            self.maparray   = params['maparray0']  
            self.Gr         = params['rgrid0']  
            self.nbins      = params['bound_nbins']

        elif hamtype == "prop":
            self.Nbas       = params['Nbas']
            self.maparray   = params['maparray'] 
            self.Gr         = params['rgrid']
            self.nbins      = params['prop_nbins']
        else:
            raise ValueError("Incorrect format type for the Hamiltonian")
    
    @staticmethod
    def filter_mat(mat,filter):
        """
        Removes elements of a sparse matrix whose magnitude is smaller than a given threshold.

        Args:
            mat: array (numpy or sparse)
            filter (float): threshold value

        Returns: array (numpy.ndarray or scipy.sparse) , dtype = mat.dtype, shape = mat.shape

        """
        nonzero_mask        = np.array(np.abs(mat[mat.nonzero()]) < filter)[0]
        rows                = mat.nonzero()[0][nonzero_mask]
        cols                = mat.nonzero()[1][nonzero_mask]
        mat[rows, cols]     = 0
        return mat

    
    @staticmethod
    def calc_mat_density(mat):
        dens =  mat.getnnz() / (mat.shape[0]*mat.shape[1])
        print("The density of the sparse matrix = " + str(dens*100) + " %")

    
    def call_eigensolver(self,A):
        if self.params['ARPACK_enr_guess'] == None:
            print("No eigenvalue guess defined")
        else:
            self.params['ARPACK_enr_guess'] /= constants.au_to_ev


        if self.params['ARPACK_which'] == 'LA':
            print("using which = LA option in ARPACK: changing sign of the Hamiltonian")
            
            #B = A.copy()
            #B = B-A.getH()
            #print(B.count_nonzero())
            #if not B.count_nonzero()==0:
            #    raise ValueError('expected symmetric or Hermitian matrix!')

            enr, coeffs = eigsh(    -1.0 * A, k = self.params['num_ini_vec'], 
                                    which=self.params['ARPACK_which'] , 
                                    sigma=self.params['ARPACK_enr_guess'],
                                    return_eigenvectors=True, 
                                    mode='normal', 
                                    tol = self.params['ARPACK_tol'],
                                    maxiter = self.params['ARPACK_maxiter'])
            enr *= -1.0
            enr = np.sort(enr)

            coeffs_sorted = np.copy(coeffs)

            for i in range(self.params['num_ini_vec']):
                coeffs_sorted[:,i] = coeffs[:,self.params['num_ini_vec']-i-1]
            coeffs = coeffs_sorted


        else:
            enr, coeffs = eigsh(    A, k = self.params['num_ini_vec'], 
                                    which=self.params['ARPACK_which'] , 
                                    sigma=self.params['ARPACK_enr_guess'],
                                    return_eigenvectors=True, 
                                    mode='normal', 
                                    tol = self.params['ARPACK_tol'],
                                    maxiter = self.params['ARPACK_maxiter'])

        #print(coeffs.shape)
        #print(enr.shape)
        #sort coeffs

        return enr, coeffs
      
    @staticmethod
    @jit(nopython=True,parallel=False,fastmath=False)
    def gen_klist_jit(Nbas,maparray):
        """Calculate a list of all KEO indices, for which matrix elements can be non-zero 
        
        Returns: list
            klist: a list of indices
        """
        klist = []

        for p1 in range(Nbas):
            for p2 in range(p1, Nbas):
                if maparray[p1,3] == maparray[p2,3] and maparray[p1,4] == maparray[p2,4]:
                    if maparray[p1,0] == maparray[p2,0] or maparray[p1,0] == maparray[p2,0] - 1 or maparray[p1,0] == maparray[p2,0] + 1: 
                        klist.append([ maparray[p1,0], maparray[p1,1], maparray[p2,0], \
                        maparray[p2,1], maparray[p2,3], p1, p2 ])

        return klist

    def call_gen_klist_jit(self,Nbas):
        maparray = np.asarray(self.maparray, dtype = int)
        klist = self.gen_klist_jit(Nbas,maparray)
        return klist

    def gen_klist(self):
        """
        Generates a list of non-zero indices in the KEO matrix 

        Returns: list
                klist: 1D list
                    list [i1,n1,i2,n2,l2,p1,p2]
        """
        maparray = np.asarray(self.maparray, dtype = int)
        klist = []

        for p1 in range(self.Nbas):
            for p2 in range(p1, self.Nbas):
                if maparray[p1,3] == maparray[p2,3] and maparray[p1,4] == maparray[p2,4]:
                    if maparray[p1,0] == maparray[p2,0] or maparray[p1,0] == maparray[p2,0] - 1 or maparray[p1,0] == maparray[p2,0] + 1: 
                        klist.append([ maparray[p1,0], maparray[p1,1], maparray[p2,0], \
                        maparray[p2,1], maparray[p2,3], p1, p2 ])

        return klist

    def build_keo(self):

        nlobs   = self.params['bound_nlobs'] 
        lmax    = self.params['bound_lmax']
        nlobs   = self.params['bound_nlobs']
        nbins   = self.nbins
    
        Nang    = (lmax+1)**2
        Nu      = (nlobs-1) * Nang

        x       = self.params['x']
        w       = self.params['w']

        #patching: (to consider/exclude)

            #w[0] *= 2.0
            #w[nlobs-1] *=2.0

            #print(w)
            #exit()
            #w /= np.sum(w[:])
            #w *= 0.5 * params['bound_binw']
            #scaling to quadrature range
            #x *= 0.5 * params['bound_binw']

        """ Build D-matrix """
        DMAT = self.BUILD_DMAT(x,w)

        """ Build J-matrix """
        JMAT  = self.BUILD_JMAT(DMAT,w)

        """
        plot_mat(JMAT)
        plt.spy(JMAT, precision=params['sph_quad_tol'], markersize=5, label="J-matrix")
        plt.legend()
        plt.show()
        """

        """ Build KD, KC matrices """
        KD  = self.BUILD_KD(JMAT,w,nlobs) / (0.5 * self.params['bound_binw'])**2 #### !!!! check
        KC  = self.BUILD_KC(JMAT,w,nlobs) / (0.5 * self.params['bound_binw'])**2
        
        """
        plot_mat(KD)
        plt.spy(KD, precision=params['sph_quad_tol'], markersize=5, label="KD")
        plt.legend()
        plt.show()
        """
        
        #inflate KD, KC and CFE
        KD_infl,KC_infl = self.inflate(KD,KC)
        CFE             = self.build_CFE(self.Gr)

        if self.params['keo_calc_method'] == "klist":

            start_time  = time.time()
            klist       = self.call_gen_klist_jit(self.Nbas)
            end_time    = time.time()
            print("jit's first run: time for construction of the klist: " +  str("%10.3f"%(end_time-start_time)) + "s")

            klist       = np.asarray(klist, dtype=int)
            keomat      = self.fillup_keo_klist(KD,KC,self.Gr,klist)


        elif self.params['keo_calc_method'] == "slices":

            keomat      = self.fillup_keo_lil_np(KD_infl,KC_infl,CFE)

        elif self.params['keo_calc_method'] == "vector":

            keomat      = self.fillup_keo_csr(KD_infl,KC_infl,CFE,nbins,Nu,Nang)
        
        else:
            raise ValueError("Incorrect method name for calculating the KEO")

    
        #print("KEO difference:")
        #print((keomat_lil-keomat).toarray())

        #print(keomat_lil.get_shape())
        #print(keomat.get_shape())
 
        #enr1,coeffs =self.call_eigensolver(keomat.toarray())
        #enr2,coeffs =self.call_eigensolver(keomat2.toarray())
        #print(enr1*constants.au_to_ev)
        #print(enr2*constants.au_to_ev )
        #print("eigenvalues difference:")
        #print(enr1-enr2)


        #plot_mat(keomat,show=False,save=True,name="KEO")
        
        #plt.spy(keomat, precision=1e-8, markersize=5, label="KEO")
        #plt.legend()
        #plt.show()
        #exit()

        print("KEO:\n")
        self.calc_mat_density(keomat)

        #print size of KEO matrix
        keo_csr_size = keomat.data.size/(1024**2)
        print('Size of the sparse KEO csr_matrix: '+ '%3.2f' %keo_csr_size + ' MB')

        return  keomat

    def build_CFE(self,Gr):
        """
        Builds the vector of the centrifugal energy operator
            CFE = l(l+1)/r^2

        Args:
            Gr: (1D numpy array): radial grid 

        Returns: 
            CFE: (1D numpy array): values of the centrifugal energy operator at grid points

        """

        lmax    = self.params['bound_lmax']
        CFE     = np.zeros(self.Nbas, dtype = float)

        for ipoint in range(Gr.shape[0]):
            for l in range(lmax+1):
                for m in range(-l,l+1):
                    p = ipoint*(lmax+1)**2 + l*(l+1) + m
                    CFE[p] = l*(l+1)/Gr[ipoint]**2 #- 2.0/Gr[ipoint]

        return CFE

    def inflate(self,KD,KC):
        """
        Inflates generic KD and KC matrices into the angular-radial basis in a single bin.

        Args: 
            KD (2D numpy array, shape=(nlobs-1,nlobs-1)): the t_ij array in the KEO
            KC (1D numpy array, shape=(1,nlobs-1)): the u_ij array in the KEO coupling adjacent bins

        Returns: tuple
                KD0: numpy 2D array, shape = ( (nlobs-1)*(lmax+1)**2 , (nlobs-1)*(lmax+1)**2 ) 
                    Inflated KD matrix
                KC0: numpy 2D array, shape = ( (lmax+1)**2, (nlobs-1)*(lmax+1)**2 ) 
                    Inflated KC matrix
        """

        lmax    = self.params['bound_lmax']
        nlobs   = self.params['bound_nlobs'] 

        KD0 = np.zeros( ((nlobs-1)*(lmax+1)**2, (nlobs-1)*(lmax+1)**2), dtype=float)
        KC0 = np.zeros( ((lmax+1)**2, (nlobs-1)*(lmax+1)**2), dtype=float)


        for n1 in range(nlobs-1):
            for n2 in range(nlobs-1):
                #put the indices below into a list and slice for vectorization
                for l in range(lmax+1):
                    for m in range(-l,l+1):
                        KD0[n1*(lmax+1)**2+l*(l+1)+m, n2*(lmax+1)**2+l*(l+1)+m] = KD[n1,n2]


        for n1 in range(nlobs-1):
            for l in range(lmax+1):
                for m in range(-l,l+1):
                    KC0[l*(l+1)+m,n1*(lmax+1)**2+l*(l+1)+m] = KC[n1]

        return KD0, KC0

    def fillup_keo_lil_np(self,KD0,KC0,CFE):
        """
        Fill up the KEO matrix with pre-calculated blocks of matrix elements. Use slicing of lil sparse matrix.

        Args:
            KD0 (2D numpy array): inflated KD matrix
            KC0 (2D numpy array): inflated KC matrix
            CFE (1D numpy array): The centrifugal term

        Returns:
            keomat (2D sparse csr array): the KEO operator matrix in csr format

        """
        lmax    = self.params['bound_lmax']
        nlobs   = self.params['bound_nlobs']
        nbins   = self.nbins
        #print("Nbas = " + str(self.Nbas))
        #print("predicted Nbas = " + str(int(((self.nbins)*(nlobs-1)-1)*(lmax+1)**2)))

        #keomat_diag = sparse.diags(CFE,0,(self.Nbas, self.Nbas), dtype=float, format="lil")
        keomat = sparse.lil_matrix((self.Nbas, self.Nbas), dtype=float)

        start_ind_kd    = []
        end_ind_kd      = []

        start_ind_col_kc    = []
        end_ind_col_kc      = []

        start_ind_row_kc    = []
        end_ind_row_kc      = []


        Nang = (lmax+1)**2
        Nu = (nlobs-1) * Nang

        #bins = 0,1,...,nbins-2
        for ibin in range(nbins-2):
            
            start_ind_kd.append( ibin*Nu )
            end_ind_kd.append( (ibin+1)*Nu-1 ) #checked manually. OK.

            start_ind_row_kc.append(( (ibin+1)*(nlobs-1)-1)*Nang )
            end_ind_row_kc.append( (ibin+1)*Nu-1)

            start_ind_col_kc.append( (ibin+1)*Nu )
            end_ind_col_kc.append( (ibin+2)*Nu - 1 )


            #print("start_ind_kd for i = " + str(ibin) + " = " + str(start_ind_kd[ibin]))
            #print("end_ind_kd for i = " + str(ibin) + " = " + str(end_ind_kd[ibin]))

            #print("start_ind_row_kc for i = " + str(ibin) + " = " + str(start_ind_row_kc[ibin]))
            #print("end_ind_row_kc for i = " + str(ibin) + " = " + str(end_ind_row_kc[ibin]))

            #print("start_ind_col_kc for i = " + str(ibin) + " = " + str(start_ind_col_kc[ibin]))
            #print("end_ind_col_kc for i = " + str(ibin) + " = " + str(end_ind_col_kc[ibin]))


            keomat[ start_ind_kd[ibin]:end_ind_kd[ibin]+1, start_ind_kd[ibin]:end_ind_kd[ibin]+1 ] = KD0
            
            keomat[ start_ind_row_kc[ibin]:end_ind_row_kc[ibin]+1, start_ind_col_kc[ibin]:end_ind_col_kc[ibin]+1 ] = KC0
            keomat[ start_ind_col_kc[ibin]:end_ind_col_kc[ibin]+1, start_ind_row_kc[ibin]:end_ind_row_kc[ibin]+1 ] = KC0.T

        #last two bins are special

        #one to last has bridges but shorter by one grid point:

        start_ind_kd.append( (nbins-2)*Nu )
        end_ind_kd.append( (nbins-1)*Nu-1 ) #checked manually. OK.

        start_ind_row_kc.append( ((nbins-1)*(nlobs-1)-1)*Nang )
        end_ind_row_kc.append( (nbins-1)*Nu-1)

        start_ind_col_kc.append( (nbins-1)*Nu )
        end_ind_col_kc.append( (nbins * (nlobs - 1) -1)*Nang -1 )
    

        #print("start_ind_kd for i = " + str(nbins-2) + " = " + str(start_ind_kd[nbins-2]))
        #print("end_ind_kd for i = " + str(nbins-2) + " = " + str(end_ind_kd[nbins-2]))

        #print("start_ind_row_kc for i = " + str(nbins-2) + " = " + str(start_ind_row_kc[nbins-2]))
        #print("end_ind_row_kc for i = " + str(nbins-2) + " = " + str(end_ind_row_kc[nbins-2]))

        #print("start_ind_col_kc for i = " + str(nbins-2) + " = " + str(start_ind_col_kc[nbins-2]))
        #print("end_ind_col_kc for i = " + str(nbins-2) + " = " + str(end_ind_col_kc[nbins-2]))

        keomat[ start_ind_kd[nbins-2]:end_ind_kd[nbins-2]+1, start_ind_kd[nbins-2]:end_ind_kd[nbins-2]+1 ] = KD0
        
        keomat[ start_ind_row_kc[nbins-2]:end_ind_row_kc[nbins-2]+1, start_ind_col_kc[nbins-2]:end_ind_col_kc[nbins-2]+1 ] = KC0[:,:(nlobs-2)*(lmax+1)**2]
        keomat[ start_ind_col_kc[nbins-2]:end_ind_col_kc[nbins-2]+1, start_ind_row_kc[nbins-2]:end_ind_row_kc[nbins-2]+1 ] = KC0.T[:(nlobs-2)*(lmax+1)**2,:] #Checked
    
        #last bin (no bridges)
      
        start_ind_kd.append( (nbins-1)*(nlobs-1)*Nang )
        end_ind_kd.append( ( nbins*(nlobs-1)-1)*Nang-1 )

        keomat[ start_ind_kd[nbins-1]:end_ind_kd[nbins-1]+1, start_ind_kd[nbins-1]:end_ind_kd[nbins-1]+1 ] = KD0[:(nlobs-2)*(lmax+1)**2,:(nlobs-2)*(lmax+1)**2]
        
        keomat = keomat.tocsr()
        keomat /=2.0
        keomat +=  sparse.diags(CFE,0,(self.Nbas, self.Nbas), dtype=float, format="csr") /2.0

        return keomat

    def fillup_keo_klist(self,KD,KC,Gr,klist):

        nlobs = self.params['bound_nlobs']

        if self.params['hmat_format'] == 'numpy_arr':    
            keomat =  np.zeros((self.Nbas, self.Nbas), dtype=float)
        elif self.params['hmat_format'] == 'sparse_csr':
            keomat = sparse.csr_matrix((self.Nbas, self.Nbas), dtype=float)
        else:
            raise ValueError("Incorrect format type for the Hamiltonian")

        for i in range(klist.shape[0]):
            if klist[i,0] == klist[i,2]:
                #diagonal blocks
                #print(klist[i,1],klist[i,3])
                keomat[ klist[i,5], klist[i,6] ] += KD[ klist[i,1] , klist[i,3]  ] #basis indices start from 1. But Kd array starts from 0 although its elems correspond to basis starting from n=1.

                if klist[i,1] == klist[i,3]:
                    rin = Gr[ klist[i,0]*(nlobs-1)+ klist[i,1]  ] #note that grid contains all points, including n=0. Note that i=0,1,2,... and n=1,2,3... in maparray
                    #print(rin)
                    keomat[ klist[i,5], klist[i,6] ] +=  float(klist[i,4]) * ( float(klist[i,4]) + 1) / rin**2 

            elif klist[i,0] == klist[i,2] - 1: # i = i' - 1
                #off-diagonal blocks
                if klist[i,1] == nlobs - 1 : #last n
                    # u_bs and u_bb:
                    keomat[ klist[i,5], klist[i,6] ] += KC[ klist[i,3]  ]
                               
        return keomat

    def fillup_keo_csr(self,KD0,KC0,CFE,nbins,Nu,Nang):
        """Create compressed sparse-row matrix from copies of the generic inflated KD and KC matrices + the CFE term.

        Args:
            KD0 (2D numpy array): inflated KD matrix
            KC0 (2D numpy array): inflated KC matrix
            CFE (1D numpy array): The centrifugal term
            nbins (float): number of bins
            Nu (float): number of points per block (Nang*(nlobs-1))
            Nang (float): size of the angular basis

        Returns:
            keomat (2D sparse csr array): the KEO operator matrix in csr format

        """
        #form the block-diagonal part
        KD_sparse = sparse.csr_matrix(KD0)
        KD_seq = [ KD_sparse for i in range(nbins)]
        K0 = sparse.block_diag(KD_seq,format='csr')

        KC_sparse = sparse.lil_matrix(np.zeros((Nu,Nu)))
        KC_sparse[Nu-Nang:Nu,:] = KC0
        KC_sparse = KC_sparse.tocsr()
        KC_seq = [ KC_sparse for i in range(nbins-1)]
        K0b = sparse.block_diag(KC_seq,format='csr')

        K0b2 = sparse.vstack((K0b,np.zeros((Nu,(nbins-1)*Nu))))  
        K0b3 = sparse.hstack((np.zeros((nbins*Nu,Nu)),K0b2))  
        keomat = K0b3+K0b3.transpose()+K0
        keomat.resize((nbins*Nu-Nang,nbins*Nu-Nang))

        #print("Nbas = " + str(self.Nbas))
        #print("Nbas inferred from KEO construction = " + str(nbins*Nu-Nang))
        keomat /= 2.0
        keomat +=  sparse.diags(CFE,0,(self.Nbas,self.Nbas), dtype=float, format="csr") /2.0

        return keomat

    def BUILD_DMAT(self,x,w):

        N       = x.shape[0]
        DMAT    = np.zeros( ( N , N ), dtype=float)
        Dd      = np.zeros( N , dtype=float)

        for n in range(N):
            for mu in range(N):
                if mu != n:
                    Dd[n] += (x[n]-x[mu])**(-1)

       

        for n in range(N):
            DMAT[n,n] += Dd[n]/np.sqrt(w[n])

            for k in range(N):
                if n != k: 
                    DMAT[k,n]  =  1.0 / ( np.sqrt(w[n]) * (x[n]-x[k]) )

                    for mu in range(N):
                        if mu != k and mu != n:
                            DMAT[k,n] *= (x[k]-x[mu])/(x[n]-x[mu])

        #print(Dd)

        #plot_mat(DMAT)
        #plt.spy(DMAT, precision=params['sph_quad_tol'], markersize=5, label="D-matrix")
        #plt.legend()
        #plt.show()
        #exit()
        return DMAT

    def BUILD_JMAT(self,D,w):

        wdiag = np.zeros((w.size,w.size), dtype = float)
        for k in range(w.size):
            wdiag[k,k] = w[k] 
        DT = np.copy(D)
        DT = DT.T

        return multi_dot([DT,wdiag,D])
        #return np.dot( np.dot(DT,wdiag), D )

    def BUILD_KD(self,JMAT,w,N): #checked 1 May 2021
        """
        Calculates KD in coupled basis

        Returns: numpy array
                KD: numpy 1D array



        """
        
        Wb = 1.0 / np.sqrt( (w[0] + w[N-1]) ) #+ w[N-1]
        
        Ws = np.zeros(len(w), dtype = float)
        Ws = 1.0 / np.sqrt(w)

        KD = np.zeros( (N-1,N-1), dtype=float)
        #elements below are invariant to boundary quadrature scaling
        #b-b:
        KD[N-2,N-2] = Wb * Wb * ( w[N-1] * JMAT[N-1, N-1] + w[0] * JMAT[0 , 0]  )


        #b-s:
        for n2 in range(0,N-2):
            KD[N-2, n2] = np.sqrt(w[N-1]) * Wb *  JMAT[N-1, n2 + 1] #checked Ws[n2 + 1] *

        #s-b:
        for n1 in range(0,N-2):
            KD[n1, N-2] = np.sqrt(w[N-1]) * Wb *  JMAT[n1 + 1, N-1] #Ws[n1 + 1] *

        #s-s:
        for n1 in range(0, N-2):
            for n2 in range(0, N-2):
                KD[n1,n2] = JMAT[n1 + 1, n2 + 1] # Ws[n1 + 1] * Ws[n2 + 1] *  #checked. Note the shift between J-matrix and tss or Kd matrices.

        return KD  #Revisied and modified (perhaps to an equivalent form on 28 Oct 2021)

    def BUILD_KC(self,JMAT,w,N):
        Wb = 1.0 / np.sqrt( (w[0]+ w[N-1] ) ) #+ w[N-1]
        
        Ws = np.zeros(len(w), dtype = float)
        Ws = 1.0 / np.sqrt(w)

        KC = np.zeros( (N-1), dtype=float)

        #b-b:
        KC[N-2] = Wb * Wb * JMAT[0, N-1] * np.sqrt(w[0] * w[N-1]) #invariant to quadrature weights scaling

        #b-s:
        for n2 in range(0, N-2):
            KC[n2] = Wb * np.sqrt(w[0]) * JMAT[0, n2 + 1] #here we exclude n'=0 which is already included in the def. of bridge function
        #changed 0->N-1 above

        return KC #checked 1 May 2021. Revisied and modified on 28 Oct 2021


    @staticmethod
    #@jit( nopython=True, parallel=False, cache = jitcache, fastmath=False) 
    def calc_potmat_anton_jit( vLM, vlist, tjmat):
        """Obsolete function.
            
            Calculate indices and values for the non-zero elements of the potential energy matrix for the chiralium potential.

        Arguments:
            vLM (3D numpy array, shape=(Nr,Lmax+1,2Lmax+1)): array of values of the radial potential energy expanded in spherical harmonics basis
            vlist (2D numpy array, shape=(No.nonzero elems,7)): list of non-zero matrix element
            tjmat (6D numpy array): matrix elements of the respective spherical harmonics operators in the spherical harmonics basis

        Returns: tuple
            pot (list, shape=(Nbas)): a list of values
            ind (list): a list of indices
        """
        pot = []
        potind = []
        Lmax = vLM.shape[1]
        for p in range(vlist.shape[0]):
          
            v = 0.0+1j*0.0
            for L in range(Lmax):
                for M in range(-L,L+1):
                
                    v += vLM[vlist[p,0],L,L+M] * tjmat[vlist[p,1], L, vlist[p,3], L + M, vlist[p,1] + vlist[p,2],vlist[p,3] + vlist[p,4]]
            pot.append( [v] )
            potind.append( [ vlist[p,5], vlist[p,6] ] )

        return pot, potind


    #@jit( nopython=True, parallel=False, cache = jitcache, fastmath=False) 
    def gen_vlist(self,maparray, Nbas, map_type):
        #create a list of indices for matrix elements of the potential
        vlist = []

        #needs speed-up: convert maparray to numpy array and vectorize, add numpy

        if map_type == 'DVR':
            for p1 in range(Nbas):
                for p2 in range(Nbas):
                    if maparray[p1][2] == maparray[p2][2]: 
                        vlist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], \
                            maparray[p2][3], maparray[p2][4], p1, p2  ])

        elif map_type == 'SPECT':
            for p1 in range(Nbas):
                for p2 in range(p1, Nbas):
                    if maparray[p1][2] == maparray[p2][2]:
                        vlist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], maparray[p2][3], maparray[p2][4] ])
        return vlist

    def map_tjmat(self,tjmat):
        """Map the 6D tjmat onto a practical 3D numpy array.

        Args:
            tjmat (6D numpy array): matrix elements of the respective spherical harmonics operators in the spherical harmonics basis

        Returns: array
            tjmats (3D numpy array, shape=(Nang,Nang,Nr)): tjmat reduced to 3 dimensions
        """
        lmax = self.params['bound_lmax']
        Lmax = self.params['multi_lmax']

        Nang = (lmax+1)**2
        Nmulti = (Lmax+1)**2

        #print("Nang = " + str(Nang))
        #print("Nmulti = " + str(Nmulti))

        tjmats = np.zeros((Nang,Nang,Nmulti),dtype=float)
        #tjmats2 = np.zeros((Nang,Nang,Nmulti),dtype=float)

        """
        start_time = time.time()
        #using local basis indices with counters
        s1 = -1
        for l1 in range(lmax):
            for m1 in range(-l1,l1+1):
                s1+=1
                s2=-1
                for l2 in range(lmax):
                    for m2 in range(-l2,l2+1):
                        s2+=1
                        imult = -1
                        for L in range(Lmax):
                            for M in range(-L,L+1):
                                imult +=1
                                tjmats1[s1,s2,imult] = tjmat[l1,L,l2,l1+m1,L+M,l2+m2]
        
        #using local basis indices without counters
        end_time = time.time()
        print("Time for the construction of tjmats using counters: " +  str("%10.3f"%(end_time-start_time)) + "s")
        
        """
        start_time = time.time()
        #using local basis indices without counters: a bit slower
        for l1 in range(lmax+1):
            for m1 in range(-l1,l1+1):

                for l2 in range(lmax+1):
                    for m2 in range(-l2,l2+1):
       
                        for L in range(Lmax+1):
                            for M in range(-L,L+1):

                                tjmats[l1*(l1+1)+m1,l2*(l2+1)+m2,L*(L+1)+M] = tjmat[l1,L,l2,l1+m1,L+M,l2+m2]
        
        end_time = time.time()
        print("Time for the construction of tjmats withouts counters: " +  str("%10.3f"%(end_time-start_time)) + "s")
        
        #print(np.allclose(tjmats1,tjmats2))
        #exit()
        return tjmats


    def map_vmats(self,vLM,Nr):
        """Map the 3D vLM array onto a practical 2D numpy array.

        Args:
            vLM (3D numpy array, shape=(Nr,Lmax+1,2Lmax+1)): array of values of the radial potential energy expanded in spherical harmonics basis
        
        Returns: array
            vmats (2D numpy array, shape=(Nr,Nmulti)): Nmulti = (Lmax+1)^2
        """
        Lmax = self.params['multi_lmax']
        Nmulti = (Lmax+1)**2

        vmats = np.zeros((Nr,Nmulti),dtype=complex)

        for L in range(Lmax+1):
            for M in range(-L,L+1):
                vmats[:,L*(L+1)+M] = vLM[:,L,L+M]
        
        return vmats


    def calc_vxi(self,vmat,tmats):
        """Calculate a block of the potential energy matrix for a single grid point.

        Arguments:
            vmats (1D numpy array, shape=(Nmulti)): Nmulti = (Lmax+1)^2, values of the radial potential at a single grid point.
            tmats (3D numpy array, shape=(Nang,Nang,Nmulti)): tjmat reduced to 3 dimensions
        
        Returns: numpy.ndarray, dtype = complex, shape = (Nang,Nang)
            vxi: 2D numpy array keeping the matrix elements of the potential energy for a given radial grid point
        """
        Nang = tmats.shape[0]
        Nmulti = vmat.shape[0]
        #print("Nmulti in calc_vxi = " +str(Nmulti))
        vxi = np.zeros((Nang,Nang), dtype = complex)

        for imulti in range(Nmulti):
            vxi += vmat[imulti] * tmats[:,:,imulti]
        #vxi = np.dot(vmat[ipoint,:],tmats[,,:])

        return vxi

    def build_potmat(self,grid_euler=[0,0,0],irun=0):
        """Driver routine for the calculation of the potential energy matrix.

            Here it is decided which routine to call. We have a choice of different routines depending on the molecule and the type of method used to calculate matrix elements.

        Args: tuple
            grid_euler: full grid or Euler angles
            irun:   id of the run when doing orientation-averaging

        Returns: sparse array
            potmat: sparse array, dtype = complex, shape = (Nbas, Nbas)      
        
        """
  
        if self.params['molec_name'] == "chiralium":

            if self.params['matelem_method'] == "analytic":

                print("Chiralium: building the potential energy matrix elements using multipole expansion of the potential and analytic integrals")
                
                if self.params['esp_mode'] == "anton":
                    potmat = self.build_potmat_chiralium_anton(grid_euler,irun)
                else:
                    raise NotImplementedError("ESP " + str(self.params['esp_mode']) + " not implemented for this molecule and matelem_method" )

            elif self.params['matelem_method'] == "lebedev":
         
                print("Chiralium: building the potential energy matrix elements using multipole expansion of the potential and numerical (quadratures) integrals")
                if self.params['esp_mode'] == "anton":
                    potmat = self.build_potmat_chiralium_anton_num(grid_euler,irun)
                else:
                    raise NotImplementedError("ESP " + str(self.params['esp_mode']) + " not implemented for this molecule and matelem_method" )
                
        elif self.params['molec_name'] == "h":

            if self.params['matelem_method'] == "analytic":
                
                if self.params['esp_mode'] == "analytic":
                    print("Hydrogen atom: building the potential energy matrix elements using analytic potential and analytic integrals")
                else:
                    raise NotImplementedError("ESP " + str(self.params['esp_mode']) + " not implemented for this molecule and matelem_method" )
        else: 
            print(str(self.params['molec_name']) + ": building the potential energy matrix elements numerical potential and numerical (quadratures) integrals")
        
        return potmat

    def build_potmat_chiralium_anton(self,grid_euler,irun):
        """Build the potential energy matrix using the analytic method with the ESP read from files provided by A. Artemyev.

            See theory docs for more detailes.

            Returns: sparse matrix
                potmat sparse matrix, dtype = complex, shape = (Nbas,Nbas)
                    potential energy matrix

            .. note:: For this function to work a set of ESP files in ./potential/ directory must be provided.

            Examples:   
                Below are given example structures of the potential energy matrix:

                .. figure:: _images/potmat_example_zoom1.png
                    :height: 400
                    :width: 400
                    :align: center

                .. figure:: _images/potmat_example_zoom2.png
                    :height: 400
                    :width: 400
                    :align: center

                .. figure:: _images/potmat_example_zoom3.png
                    :height: 400
                    :width: 400
                    :align: center          

        """
        Nr = self.Gr.ravel().shape[0]
        # Read the potential partial waves on the grid
        start_time = time.time()
        vLM,rgrid_anton = potential.read_potential_anton(self.params,Nr,False)
        end_time = time.time()
        print("Time for the construction of the potential partial waves: " +  str("%10.3f"%(end_time-start_time)) + "s")

        # Build an array of 3-j symbols
        tjmat       = gen_tjmat(self.params['bound_lmax'],self.params['multi_lmax'])

        tmats = self.map_tjmat(tjmat)

        vmats = self.map_vmats(vLM,Nr)

        # This part can probably be done in parallel
        potarr = []
        for ipoint in range(Nr):
            vxi = self.calc_vxi(vmats[ipoint],tmats)
            potarr.append(vxi)

        potmat = sparse.csr_matrix((self.Nbas,self.Nbas),dtype=complex)

        # build potmat from block-diagonal parts in each bin
        potmat = sparse.block_diag(potarr)
        
        #plot the matrix:
        #plot_mat(potmat,1.0,show=True,save=True,name="potmat_example",path="./")
        #plt.spy(potmat.todense(),precision=1e-3,color='b', markersize=5)
        #plt.show()
        #exit()
        return potmat


    def build_potmat_standard(self,grid_euler=[0,0,0],irun=0):
        """ 
        Driver routine for the calculation of the potential energy matrix.
        We keep it temporarily.
    
        """
        #full
        Nr = self.Gr.ravel().shape[0]
        #bound
        #Nr = (self.params['bound_nlobs']-1) * self.params['bound_nbins']  
        # 1. Construct vlist
        start_time = time.time()
        vlist = self.gen_vlist( self.maparray, self.Nbas, self.params['map_type'] )
        vlist = np.asarray(vlist,dtype=int)
        end_time = time.time()
        print("Time for the construction of vlist: " +  str("%10.3f"%(end_time-start_time)) + "s")

        #klist = mapping.GEN_KLIST(maparray, Nbas, params['map_type'] )

        #klist = np.asarray(klist)
        #vlist = np.asarray(vlist)

        #print(klist.shape[0])
        #print(vlist.shape[0])

        #print(np.vstack((klist,vlist)))

        #exit()

        # 2. Read the potential partial waves on the grid
        start_time = time.time()
        vLM,rgrid_anton = potential.read_potential(self.params,Nr,False)
        end_time = time.time()
        print("Time for the construction of the potential partial waves: " +  str("%10.3f"%(end_time-start_time)) + "s")
        
        #print(rgrid_anton-self.Gr)
        #exit()
        #test_vlm_symmetry(vLM)

        #print("Maximum imaginary part of the potential:")
        #print(vLM.imag.max())
        #print(vLM.imag.min())
        #exit()

        # 3. Build array of 3-j symbols
        tjmat       = gen_tjmat(self.params['bound_lmax'],self.params['multi_lmax'])
        #tjmat       = gen_tjmat_quadpy(params['bound_lmax'],params['multi_lmax']) #for tjmat generated with quadpy
        #tjmat       = gen_tjmat_leb(params['bound_lmax'],params['multi_lmax']) #for tjmat generated with generic lebedev



        #print("tjmat-tjmatrot")
        #print(np.allclose(tjmat,tjmat_rot))
        #print("tjmatrot")
        #print(tjmat_rot)
        #exit()

        """ Testing of grid compatibility"""
        """
        print(Gr.ravel().shape)
        print(rgrid_anton.shape) #the two grids agree
        if np.allclose(Gr.ravel(), rgrid_anton, rtol=1e-6, atol=1e-6)==False:
            raise ValueError('the radial grid does not match the grid of the potential')
        print(max(abs(rgrid_anton-Gr.ravel())))
        exit()
        """    
        start_time = time.time()
        # 4. sum-up partial waves
        if self.params['N_euler'] == 1:
            potmat0, potind = self.calc_potmat_anton_jit( vLM, vlist, tjmat )
        else:
            tjmat_rot       = rotate_tjmat(grid_euler,irun,tjmat)
            potmat0, potind = self.calc_potmat_anton_jit( vLM, vlist, tjmat_rot )

        end_time = time.time()
        print("Time for calculation of the potential matrix elements: " +  str("%10.3f"%(end_time-start_time)) + "s")
        

        #potmat0 = np.asarray(potmat0)
        #print(potmat0[:100])
        #print("Maximum real part of the potential matrix = " + str(np.max(np.abs(potmat0.real))))
        #print("Maximum imaginary part of the potential matrix = " + str(np.max(np.abs(potmat0.imag))))
        #exit()
        # 5. Return final potential matrix
        potmat = sparse.csr_matrix((self.Nbas,self.Nbas),dtype=complex)
        potind = np.asarray(potind,dtype=int)
        #potmat0 = np.asarray(potmat0,dtype=complex)
        """ Put the indices and values back together in the Hamiltonian array"""
        #print(potmat0.shape)
        #print(self.Nbas)
        #exit()
        for ielem, elem in enumerate(potmat0):
            #print(elem[0])
            potmat[ potind[ielem,0], potind[ielem,1] ] = elem[0]
        plot_mat(potmat)
        plt.spy(potmat,precision=1e-4)
        plt.show()
        exit()
        return potmat#+potmat.getH()#-potmat.diagonal()


    def build_ham(self,keo,pot):
        """Build the bound Hamiltonian from the KEO and POT matrices provided. Return a filtered Hamiltonian. 
        
        
        """

        ham         = keo + pot
        ham_filt    = Hamiltonian.filter_mat(ham,self.params['hmat_filter'])
        
        #plt.spy(ham_bound,precision=1e-4)
        #plt.show()
        #exit()
        
        return ham_filt

    def map_tjmat_dip(self,tjmat):
        """Map the 6D tjmat onto a practical 3D numpy array.

        Args:
            tjmat (6D numpy array): matrix elements of the respective spherical harmonics operators in the spherical harmonics basis

        Returns: array
            tjmats (3D numpy array, shape=(Nang,Nang,Nr)): tjmat reduced to 3 dimensions
        """
        lmax = self.params['bound_lmax']


        Nang = (lmax+1)**2
        #print("Nang = " + str(Nang))
        #print("Nmulti = " + str(Nmulti))

        tjmats = np.zeros((Nang,Nang,3),dtype=float)
        #tjmats2 = np.zeros((Nang,Nang,Nmulti),dtype=float)

        """
        start_time = time.time()
        #using local basis indices with counters
        s1 = -1
        for l1 in range(lmax):
            for m1 in range(-l1,l1+1):
                s1+=1
                s2=-1
                for l2 in range(lmax):
                    for m2 in range(-l2,l2+1):
                        s2+=1
                        imult = -1
                        for L in range(Lmax):
                            for M in range(-L,L+1):
                                imult +=1
                                tjmats1[s1,s2,imult] = tjmat[l1,L,l2,l1+m1,L+M,l2+m2]
        
        #using local basis indices without counters
        end_time = time.time()
        print("Time for the construction of tjmats using counters: " +  str("%10.3f"%(end_time-start_time)) + "s")
        
        """
        start_time = time.time()
        #using local basis indices without counters: a bit slower
        for l1 in range(lmax+1):
            for m1 in range(-l1,l1+1):

                for l2 in range(lmax+1):
                    for m2 in range(-l2,l2+1):
       
                      for mu in range(3):
                                tjmats[l1*(l1+1)+m1,l2*(l2+1)+m2,mu] = tjmat[l1,l2,l1+m1,l2+m2,mu]
        
        end_time = time.time()
        print("Time for the construction of tjmats_dip withouts counters: " +  str("%10.3f"%(end_time-start_time)) + "s")
        
        #print(np.allclose(tjmats1,tjmats2))
        #exit()
        return tjmats



    def calc_dipxi(self,rin,tmats):
        """Calculate a block of the electric dipole interaction matrix for a single grid point.

        Args:
            vmats (1D numpy array, shape=(Nmulti)): Nmulti = (Lmax+1)^2, values of the radial potential at a single grid point.
            tmats (3D numpy array, shape=(Nang,Nang,Nr)): tjmat reduced to 3 dimensions
        
        Returns: array
            vxi (2D numpy array, shape=(Nmulti,Nmulti)): Nmulti = (Lmax+1)^2
        """
        Nang = tmats.shape[0]
        vxi = np.zeros((Nang,Nang), dtype = complex)

        vxi = rin * tmats[:,:]
 
        return vxi



    def gen_3j_dip(self,lmax,mode):
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


    def build_intmat(self):
        """ 
        Driver routine for the calculation of the dipole interaction energy matrix.

    
        """
  
        Nr = self.Gr.ravel().shape[0]

        rgrid = self.Gr.ravel()

        """precompute all necessary 3-j symbols"""
        #generate arrays of 3j symbols with 'spherical':
        start_time = time.time()
        tjmat = self.gen_3j_dip(self.params['bound_lmax'], "CG")
        end_time = time.time()
        print("Time for calculation of tjmat in dipole matrix =  " + str("%10.3f"%(end_time-start_time)) + "s")
        

        tmats = self.map_tjmat_dip(tjmat)

        if self.params['field_func_name'] == "RCPL" or self.params['field_func_name'] == "LCPL":
            tmat = tmats[:,:,2]
        elif self.params['field_func_name'] == "LP":
            tmat = tmats[:,:,1]
        else:
            raise NotImplementedError("Incorrect field name") 

        #this part can probably be done in parallel
        diparr = []
        for ipoint in range(Nr):
            dipxi = self.calc_dipxi(rgrid[ipoint],tmat)
            diparr.append(dipxi)

        intmat = sparse.csr_matrix((self.Nbas,self.Nbas),dtype=complex)

        intmat = sparse.block_diag(diparr)

        #plot_mat(intmat)
        #plt.spy(intmat+intmat.H,precision=1e-12)
        #plt.show()
        #exit()
        return (-1.0) * np.sqrt( 2.0 * np.pi / 3.0 ) * intmat

def fp(i,n,k,x):
    "calculate d/dr f_in(r) at r_ik (Gauss-Lobatto grid) "
    npts = x.size 
    if n!=k: #issue with vectorization here.
        fprime  =   (x[n]-x[k])**(-1)
        prod    =   1.0e00

        for mu in range(npts):
            if mu !=k and mu !=n:
                prod *= (x[k]-x[mu])/(x[n]-x[mu])
        fprime  *=  prod

    elif n==k:
        fprime  =   0.0
        for mu in range(npts):
                if mu !=n:
                    fprime += (x[n]-x[mu])**(-1)
    return fprime



""" ============ POTMAT0 ============ """
def BUILD_POTMAT0( params, maparray, Nbas , Gr ):

    if params['esp_mode'] == "interpolation":
        print("Interpolating electrostatic potential")
        esp_interpolant = POTENTIAL.INTERP_POT(params)

        if  params['gen_adaptive_quads'] == True:
            sph_quad_list = gen_adaptive_quads( params, esp_interpolant, Gr )
        elif params['gen_adaptive_quads'] == False and params['use_adaptive_quads'] == True:
            sph_quad_list = read_adaptive_quads(params)
        elif params['gen_adaptive_quads'] == False and params['use_adaptive_quads'] == False:
            print("using global quadrature scheme")
            sph_quad_list = [params['sph_quad_global']]

    elif params['esp_mode'] == "exact":
        if  params['gen_adaptive_quads'] == True:
            sph_quad_list = gen_adaptive_quads_exact( params,  Gr ) #checked 30 Apr 2021
        elif params['gen_adaptive_quads'] == False and params['use_adaptive_quads'] == True:
            sph_quad_list = read_adaptive_quads(params)
        elif params['gen_adaptive_quads'] == False and params['use_adaptive_quads'] == False:
            print("using global quadrature scheme")



    start_time = time.time()
    Gs = GRID.GEN_GRID( sph_quad_list, params['main_dir'] )
    end_time = time.time()
    print("Time for grid generation: " +  str("%10.3f"%(end_time-start_time)) + "s")

    if params['esp_mode'] == "interpolate":
        start_time = time.time()
        VG = POTENTIAL.BUILD_ESP_MAT( Gs, Gr, esp_interpolant, params['r_cutoff'] )
        end_time = time.time()
        print("Time for construction of ESP on the grid: " +  str("%10.3f"%(end_time-start_time)) + "s")

    elif params['esp_mode'] == "exact":
        start_time = time.time()
        VG = POTENTIAL.BUILD_ESP_MAT_EXACT(params, Gs, Gr)
        end_time = time.time()
        print("Time for construction of ESP on the grid: " +  str("%10.3f"%(end_time-start_time)) + "s")

    #if params['enable_cutoff'] == True: 
    #print() 

    start_time = time.time()
    vlist = mapping.GEN_VLIST( maparray, Nbas, params['map_type'] )
    vlist = np.asarray(vlist,dtype=int)
    end_time = time.time()
    print("Time for construction of vlist: " +  str("%10.3f"%(end_time-start_time)) + "s")
    
    """we can cut vlist  to set cut-off for the ESP"""

    if params['calc_method'] == 'jit':
        start_time = time.time()
        potmat0, potind = calc_potmat_jit( vlist, VG, Gs )
        end_time = time.time()
        print("First call: Time for construction of potential matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")

        #start_time = time.time()
        #potmat0, potind = calc_potmat_jit( vlist, VG, Gs )
        #end_time = time.time()
        #print("Second call: Time for construction of potential matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")
    """
    if params['hmat_format'] == "regular":
        potmat = convert_lists_to_regular(potmat0,potind)
    elif params['hmat_format'] == "csr":
        potmat = convert_lists_to_csr(potmat0,potind)
    """
    return potmat0, potind

def euler_rot(chi, theta, phi, xyz):
    """Rotates Cartesian vector xyz[ix] (ix=x,y,z) by an angle phi around Z,
    an angle theta around new Y, and an angle chi around new Z.
    Input values of chi, theta, and phi angles are in radians.
    """
    amat = np.zeros((3,3), dtype=np.float64)
    bmat = np.zeros((3,3), dtype=np.float64)
    cmat = np.zeros((3,3), dtype=np.float64)
    rot = np.zeros((3,3), dtype=np.float64)

    amat[0,:] = [np.cos(chi), np.sin(chi), 0.0]
    amat[1,:] = [-np.sin(chi), np.cos(chi), 0.0]
    amat[2,:] = [0.0, 0.0, 1.0]

    bmat[0,:] = [np.cos(theta), 0.0, -np.sin(theta)]
    bmat[1,:] = [0.0, 1.0, 0.0]
    bmat[2,:] = [np.sin(theta), 0.0, np.cos(theta)]

    cmat[0,:] = [np.cos(phi), np.sin(phi), 0.0]
    cmat[1,:] = [-np.sin(phi), np.cos(phi), 0.0]
    cmat[2,:] = [0.0, 0.0, 1.0]

    rot = np.transpose(np.dot(amat, np.dot(bmat, cmat)))
    xyz_rot = np.dot(rot, xyz)
    return xyz_rot

def rotate_mol_xyz(params, grid_euler, irun):
    """ Generate raw cartesian coordinates of atoms from Z-matrix,
        followed by shift to the centre of mass,
        followed by rotation by appropriate rotation matrix associated with elements of the Euler angles grid
    """

    print("generating rotated cartesian coordinates of atoms...")
    print("irun = " + str(irun))
    print("(alpha,beta,gamma) = " + str(grid_euler[irun]))

    if params['molec_name'] == "d2s":

        mol_xyz = np.zeros( (3,3), dtype = float) #
        mol_xyz_rotated = np.zeros( (3,3), dtype = float) #

        # Sx D1x D2x
        # Sy D1y D2y
        # Sz D1z D2z

        ang_au = CONSTANTS.angstrom_to_au
        # create raw cartesian coordinates from input molecular geometry and embedding
        rSD1 = params['mol_geometry']["rSD1"] * ang_au
        rSD2 = params['mol_geometry']["rSD2"] * ang_au
        alphaDSD = params['mol_geometry']["alphaDSD"] * np.pi / 180.0

        mS = params['mol_masses']["S"]    
        mD = params['mol_masses']["D"]

        mVec = np.array ( [mS, mD, mD])

        Sz = 1.1 * ang_au  #dummy value (angstrom) of z-coordinate for S-atom to place the molecule in frame of reference. Later shifted to COM.

        mol_xyz[2,0] = Sz

        mol_xyz[1,1] = -1.0 * rSD1 * np.sin(alphaDSD / 2.0)
        mol_xyz[2,1] = -1.0 * rSD1 * np.cos(alphaDSD / 2.0 ) + Sz #checked

        mol_xyz[1,2] = 1.0 * rSD2 * np.sin(alphaDSD / 2.0)
        mol_xyz[2,2] = -1.0 * rSD2 * np.cos(alphaDSD / 2.0 ) + Sz #checked


        print("raw cartesian molecular-geometry matrix")
        print(mol_xyz)

        
        print("Centre-of-mass coordinates: ")
        RCM = np.zeros(3, dtype=float)

        for iatom in range(3):
            RCM[:] += mVec[iatom] * mol_xyz[:,iatom]

        RCM /= np.sum(mVec)

        print(RCM)
    
        for iatom in range(3):
            mol_xyz[:,iatom] -= RCM[:] 

        print("cartesian molecular-geometry matrix shifted to centre-of-mass: ")
        print(mol_xyz)
        print("Rotation matrix:")
        rotmat = R.from_euler('zyz', [grid_euler[irun][0], grid_euler[irun][1], grid_euler[irun][2]], degrees=False)
    
        #ALTErnatively use
        #euler_rot(chi, theta, phi, xyz):
    
        #rmat = rotmat.as_matrix()
        #print(rmat)

        for iatom in range(3):
            mol_xyz_rotated[:,iatom] = rotmat.apply(mol_xyz[:,iatom])
            #mol_xyz_rotated[:,iatom] = euler_rot(cgrid_euler[irun][2], grid_euler[irun][1], grid_euler[irun][0], mol_xyz[:,iatom]):
        print("rotated molecular cartesian matrix:")
        print(mol_xyz_rotated)


    elif params['molec_name'] == "n2":
        
        mol_xyz = np.zeros( (3,2), dtype = float) #
        mol_xyz_rotated = np.zeros( (3,2), dtype = float) #

        #  N1x N2x
        #  N1y N2y
        #  N1z N2z

        ang_au = CONSTANTS.angstrom_to_au

        mol_xyz[2,0] = ang_au * params['mol_geometry']["rNN"] / 2.0 
        mol_xyz[2,1] =  -1.0 * mol_xyz[2,0] 

        print("Rotation matrix:")
        rotmat = R.from_euler('zyz', [grid_euler[irun][0], grid_euler[irun][1], grid_euler[irun][2]], degrees=False)
    
        for iatom in range(2):
            mol_xyz_rotated[:,iatom] = rotmat.apply(mol_xyz[:,iatom])
        print("rotated molecular cartesian matrix:")
        print(mol_xyz_rotated)


    elif params['molec_name'] == "co":
        
        mol_xyz = np.zeros( (3,2), dtype = float) # initial embedding coordinates (0 degrees rotation)
        mol_xyz_rotated = np.zeros( (3,2), dtype = float) #
        mol_xyz_MF= np.zeros( (3,2), dtype = float) #rotated coordinates: molecular frame embedding

        ang_au = CONSTANTS.angstrom_to_au

        mC = params['mol_masses']["C"]    
        mO = params['mol_masses']["O"]
        rCO = params['mol_geometry']["rCO"] 

        # C = O   in ----> positive direction of z-axis.
        #coordinates for vanishing electric dipole moment in psi4 calculations is z_C = -0.01 a.u. , z_O = 2.14 a.u
        mol_xyz[2,0] = - 1.0 * ang_au * mO * rCO / (mC + mO) #z_C
        mol_xyz[2,1] =  1.0 * ang_au * mC * rCO / (mC + mO) #z_O

        """rotation associated with MF embedding"""
        rotmatMF = R.from_euler('zyz', [0.0, params['mol_embedding'], 0.0], degrees=True)

        for iatom in range(2):
            mol_xyz_MF[:,iatom] = rotmatMF.apply(mol_xyz[:,iatom])
        print("rotated MF cartesian matrix:")
        print(mol_xyz_MF)


        print("Rotation matrix:")
        rotmat = R.from_euler('zyz', [grid_euler[irun][0], grid_euler[irun][1], grid_euler[irun][2]], degrees=False)
    
        for iatom in range(2):
            mol_xyz_rotated[:,iatom] = rotmat.apply(mol_xyz_MF[:,iatom])
        print("rotated molecular cartesian matrix:")
        print(mol_xyz_rotated)

    elif params['molec_name'] == "h":
        
        mol_xyz = np.zeros( (3,1), dtype = float) #
        mol_xyz_rotated = np.zeros( (3,1), dtype = float) #

        #  Hx
        #  Hy 
        #  Hz 

    elif params['molec_name'] == "c":
        
        mol_xyz = np.zeros( (3,1), dtype = float) #
        mol_xyz_rotated = np.zeros( (3,1), dtype = float) #

        #  Cx
        #  Cy 
        #  Cz 

    elif params['molec_name'] == "h2o":
        mol_xyz = np.zeros( (3,1), dtype = float) #
        mol_xyz_rotated = np.zeros( (3,1), dtype = float) #
    else:
        print("Warning: molecule name not found")
        mol_xyz_rotated = np.zeros( (3,1), dtype = float) 
        #exit()

    #veryfiy rotated geometry by plots
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-2.0,2.0)
    ax.set_ylim(-2.0,2.0)
    ax.set_zlim(-2.0,2.0)
    ax.scatter(mol_xyz_rotated[0,:], mol_xyz_rotated[1,:], mol_xyz_rotated[2,:])
    ax.scatter(mol_xyz[0,:], mol_xyz[1,:], mol_xyz[2,:])
    plt.show()
    exit()

    mlab.figure(1, bgcolor=(0, 0, 0), size=(350, 350))
    mlab.clf()

    # The position of the atoms
    atoms_x = mol_xyz_rotated[0,:]
    atoms_y = mol_xyz_rotated[1,:]
    atoms_z = mol_xyz_rotated[2,:]
    axes = mlab.axes(color=(0, 0, 0), nb_labels=5)
    mlab.orientation_axes()
    O = mlab.points3d(atoms_x[1:-1], atoms_y[1:-1], atoms_z[1:-1],
                    scale_factor=3,
                    resolution=20,
                    color=(1, 0, 0),
                    scale_mode='none')

    H1 = mlab.points3d(atoms_x[:1], atoms_y[:1], atoms_z[:1],
                    scale_factor=2,
                    resolution=20,
                    color=(1, 1, 1),
                    scale_mode='none')

    H2 = mlab.points3d(atoms_x[-1:], atoms_y[-1:], atoms_z[-1:],
                    scale_factor=2,
                    resolution=20,
                    color=(1, 1, 1),
                    scale_mode='none')

    # The bounds between the atoms, we use the scalar information to give
    # color
    mlab.plot3d(atoms_x, atoms_y, atoms_z, [1, 2, 1],
                tube_radius=0.4, colormap='Reds')


    mlab.show()
    """

    return mol_xyz_rotated

""" ============ POTMAT0 ROTATED ============ """
def BUILD_POTMAT0_ROT( params, maparray, Nbas , Gr, grid_euler, irun ):

    mol_xyz = rotate_mol_xyz(params, grid_euler, irun)
  
    if params['esp_mode'] == "exact":

        if  params['gen_adaptive_quads'] == True:
            start_time = time.time()
            sph_quad_list = gen_adaptive_quads_exact_rot( params,  Gr, mol_xyz, irun ) 
            end_time = time.time()
            print("Total time for construction adaptive quads: " +  str("%10.3f"%(end_time-start_time)) + "s")

        elif params['gen_adaptive_quads'] == False and params['use_adaptive_quads'] == True:
            sph_quad_list = read_adaptive_quads_rot(params,irun)

        elif params['gen_adaptive_quads'] == False and params['use_adaptive_quads'] == False:
            print("using global quadrature scheme")
            sph_quad_list = []
            xi = 0

            for i in range(Gr.shape[0]):
                for n in range(Gr.shape[1]):
                    xi +=1
                    sph_quad_list.append([i,n+1,xi,params['sph_quad_default']])
            #print(sph_quad_list)
            #exit()

    start_time = time.time()
    Gs = GRID.GEN_GRID( sph_quad_list, params['main_dir'])
    end_time = time.time()
    print("Time for generation of the Gs grid: " +  str("%10.3f"%(end_time-start_time)) + "s")

    if params['esp_mode'] == "exact":
        start_time = time.time()
        VG = POTENTIAL.BUILD_ESP_MAT_EXACT_ROT(params, Gs, Gr, mol_xyz, irun,True)
        end_time = time.time()
        print("Time for construction of the ESP on the grid: " +  str("%10.3f"%(end_time-start_time)) + "s")

    #if params['enable_cutoff'] == True: 
    #print() 

    start_time = time.time()
    vlist = mapping.GEN_VLIST( maparray, Nbas, params['map_type'] )
    vlist = np.asarray(vlist,dtype=int)
    end_time = time.time()
    print("Time for construction of vlist: " +  str("%10.3f"%(end_time-start_time)) + "s")
    
    """we can cut vlist  to set cut-off for the ESP"""

    if params['calc_method'] == 'jit':
        start_time = time.time()
        #print(np.shape(VG))
        #print(np.shape(Gs))
        potmat0, potind = calc_potmat_jit( vlist, VG, Gs )
        #print(potind)
        end_time = time.time()
        print("First call: Time for construction of potential matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")

        #start_time = time.time()
        #potmat0, potind = calc_potmat_jit( vlist, VG, Gs )
        #end_time = time.time()
        #print("Second call: Time for construction of potential matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")
    """
    if params['hmat_format'] == "regular":
        potmat = convert_lists_to_regular(potmat0,potind)
    elif params['hmat_format'] == "csr":
        potmat = convert_lists_to_csr(potmat0,potind)
    """
    return potmat0, potind


def gen_wigner_dmats(n_grid_euler, Jmax, grid_euler):

    wigner = spherical.Wigner(Jmax)
    grid_euler = grid_euler.reshape(-1,3)
    R = quaternionic.array.from_euler_angles(grid_euler)
    D = wigner.D(R)

    WDMATS = []
    for J in range(Jmax+1):
        WDM = np.zeros((2*J+1, 2*J+1, n_grid_euler), dtype=complex)
        for m in range(-J,J+1):
            for k in range(-J,J+1):
                WDM[m+J,k+J,:] = D[:,wigner.Dindex(J,m,k)]
        #print(J,WDM)
        #print(WDM.shape)

        WDMATS.append(WDM)  
    return WDMATS


def gen_tjmat(lmax_basis,lmax_multi):
    """precompute all necessary 3-j symbols for the matrix elements of the multipole moments"""

    #store in arrays:
    # 2) tjmat[l,L,l',m,M,m'] = [0,...lmax,0...lmax,0,...,m+l,...,2l] - for definition check notes
    tjmat = np.zeros( (lmax_basis+1, lmax_multi+1, lmax_basis+1,  2 * lmax_basis + 1, 2*lmax_multi + 1, 2 * lmax_basis + 1), dtype = float)
    
    #l1 - ket, l2 - bra

    #to do: impose triangularity by creating index list for gaunt coefficients

    for l1 in range(0,lmax_basis+1):
        for l2 in range(0,lmax_basis+1):
            for L in range(0,lmax_multi+1):
                for M in range(-L,L+1):
                    for m1 in range(-l1,l1+1):
                        for m2 in range(-l2,l2+1): 
                            tjmat[l1,L,l2,l1+m1,L+M,l2+m2] = np.sqrt( (2.0*float(l2)+1) * (2.0*float(L)+1) /(  (2.0*float(l1)+1) * (4.0*np.pi) ) ) * spherical.clebsch_gordan(l2,m2,L,M,l1,m1) * spherical.clebsch_gordan(l2,0,L,0,l1,0)
                            # ( (-1.0)**(-M) ) * spherical.Wigner3j( l1,L, l2, m1, M, m2) * \
                            #                        spherical.Wigner3j( l1, L ,l2, 0, 0, 0) * \
                            #                        np.sqrt( (2.0*float(l1)+1) * (2.0*float(L)+1) * (2.0*float(l2)+1) / (4.0*np.pi) ) 


    #print(spherical.Wigner3j(1, 2, 2, 1, 1, -2))
    #exit()
    #print("3j symbols in array:")
    #print(tjmat)

    return tjmat#/np.sqrt(4.0*np.pi)

def gen_tjmat_quadpy(lmax_basis,lmax_multi):
    """precompute all necessary 3-j symbols for the matrix elements of the multipole moments"""

    #store in arrays:
    # 2) tjmat[l,L,l',M,m'] = [0,...lmax,0...lmax,0,...,m+l,...,2l] - for definition check notes
    tjmat = np.zeros( (lmax_basis+1, lmax_multi+1, lmax_basis+1, 2*lmax_multi + 1,  2 * lmax_basis + 1, 2 * lmax_basis + 1), dtype = complex)
    
    myscheme = quadpy.u3.schemes["lebedev_131"]()
    """
    Symbols: 
            theta_phi[0] = theta in [0,pi]
            theta_phi[1] = phi  in [-pi,pi]
            sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) means that we 
                                                                 put the phi angle in range [0,2pi] and 
                                                                 the  theta angle in range [0,pi] as required by 
                                                                 the scipy special funciton sph_harm
    """
    for l1 in range(0,lmax_basis+1):
        for l2 in range(0,lmax_basis+1):
            for L in range(0,lmax_multi+1):
                for M in range(-L,L+1):
                    for m1 in range(-l1,l1+1):
                        for m2 in range(-l2,l2+1):

   
    
                            tjmat_re =  myscheme.integrate_spherical(lambda theta_phi: np.real(np.conjugate( sph_harm( m1, l1,  theta_phi[1]+np.pi, theta_phi[0] ) ) \
                                                                            * sph_harm( M, L, theta_phi[1]+np.pi, theta_phi[0] ) \
                                                                            * sph_harm( m2, l2, theta_phi[1]+np.pi, theta_phi[0] )) ) 

                            tjmat_im =  myscheme.integrate_spherical(lambda theta_phi: np.imag(np.conjugate( sph_harm( m1, l1,  theta_phi[1]+np.pi, theta_phi[0] ) ) \
                                                                            * sph_harm( M, L, theta_phi[1]+np.pi, theta_phi[0] ) \
                                                                            * sph_harm( m2, l2, theta_phi[1]+np.pi, theta_phi[0] )) )



                            tjmat[l1,L,l2,L+M,l1+m1,l2+m2] = tjmat_re + 1j * tjmat_im



    return tjmat#/np.sqrt(4.0*np.pi)





def gen_tjmat_leb(lmax_basis,lmax_multi):
    """precompute all necessary 3-j symbols for the matrix elements of the multipole moments"""

    #store in arrays:
    # 2) tjmat[l,L,l',M,m'] = [0,...lmax,0...lmax,0,...,m+l,...,2l] - for definition check notes
    tjmat = np.zeros( (lmax_basis+1, lmax_multi+1, lmax_basis+1, 2*lmax_multi + 1,  2 * lmax_basis + 1, 2 * lmax_basis + 1), dtype = complex)
    
    myscheme = quadpy.u3.schemes["lebedev_131"]()
    """
    Symbols: 
            theta_phi[0] = theta in [0,pi]
            theta_phi[1] = phi  in [-pi,pi]
            sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) means that we 
                                                                 put the phi angle in range [0,2pi] and 
                                                                 the  theta angle in range [0,pi] as required by 
                                                                 the scipy special funciton sph_harm
    """
    for l1 in range(0,lmax_basis+1):
        for l2 in range(0,lmax_basis+1):
            for L in range(0,lmax_multi+1):
                for M in range(-L,L+1):
                    for m1 in range(-l1,l1+1):
                        for m2 in range(-l2,l2+1):

   
    
                            tjmat_re =  myscheme.integrate_spherical(lambda theta_phi: np.real(np.conjugate( sph_harm( m1, l1,  theta_phi[1]+np.pi, theta_phi[0] ) ) \
                                                                            * sph_harm( M, L, theta_phi[1]+np.pi, theta_phi[0] ) \
                                                                            * sph_harm( m2, l2, theta_phi[1]+np.pi, theta_phi[0] )) ) 

                            tjmat_im =  myscheme.integrate_spherical(lambda theta_phi: np.imag(np.conjugate( sph_harm( m1, l1,  theta_phi[1]+np.pi, theta_phi[0] ) ) \
                                                                            * sph_harm( M, L, theta_phi[1]+np.pi, theta_phi[0] ) \
                                                                            * sph_harm( m2, l2, theta_phi[1]+np.pi, theta_phi[0] )) )



                            tjmat[l1,L,l2,L+M,l1+m1,l2+m2] = tjmat_re + 1j * tjmat_im



    return tjmat#/np.sqrt(4.0*np.pi)

""" ============ POTMAT0 ROTATED with Multipole expansion ============ """
def BUILD_POTMAT0_MULTIPOLES_ROT( params, maparray, Nbas , Gr, grid_euler, irun ):
    """ Calculate potential matrix using multipole expansion representation of 
    the electrostatic potential. Integrals are analytic. Matrix is labeled by vlist.
    
    """

    # 1. Construct vlist
    start_time = time.time()
    vlist = mapping.GEN_VLIST( maparray, Nbas, params['map_type'] )
    vlist = np.asarray(vlist)
    end_time = time.time()
    print("Time for the construction of vlist: " +  str("%10.3f"%(end_time-start_time)) + "s")
    

    # 2. Build array of multipole moments
    start_time = time.time()
    qlm = POTENTIAL.calc_multipoles(params)
    end_time = time.time()
    print("Time for the calculation of multipole moments: " +  str("%10.3f"%(end_time-start_time)) + "s")
    

    # 3. Build array of 3-j symbols
    tjmat =  gen_tjmat(params['bound_lmax'],params['multi_lmax'])

    # 3a. Build array of '1/r**l' values on the radial grid

    rlmat = np.zeros((Gr.shape[0]*Gr.shape[1],params['multi_lmax']), dtype=float)
    for L in range(params['multi_lmax']):
        rlmat[:,L] = 1.0 / Gr.ravel()**L


    # 4. Perform semi-vectorized summation
    potmat0, potind = calc_potmat_multipoles_jit( vlist, tjmat, qlm, params['multi_lmax'], rlmat )
    # 5. Return final potential matrix
    return  potmat0, potind 


def test_vlm_symmetry(vLM):
    #function to test if vLM obeys derived symmetry rules
    Lmax = vLM.shape[1]
    nxi = vLM.shape[0]

    S = np.zeros((nxi,Lmax+1,Lmax+1),dtype = complex)

    for L in range(Lmax):
        for M in range(0,L+1):
            S[:,L,M] = vLM[:,L,L+M]-(-1)**M * vLM[:,L,L-M].conj()
            #print(S[:,L,M])

            if np.allclose(S[:,L,M],0e0+1j*0e0, 1e-8, 1e-8):
                print("L = " + str(L) + ", M = " + str(M) + ": All have zero difference")
    exit()

def rotate_tjmat(grid_euler,irun,tjmat):
    
    Lmax = tjmat.shape[1] -1

    lmax = tjmat.shape[0] -1

    tjmat_rot = np.zeros( (lmax+1, Lmax+1, lmax+1, 2*Lmax + 1,  2 * lmax + 1,  2 * lmax + 1), dtype = complex)
        
    #l1 - ket, l2 - bra

    # pull wigner matrices at the given euler angle's set
    
    n_grid_euler = grid_euler.shape[0]
    print("number of euler grid points = " + str(n_grid_euler))
    print("current Euler grid point = " + str(grid_euler[irun]))

    WDMATS = gen_wigner_dmats(1, Lmax, grid_euler[irun])

    #print(grid_euler[irun])

    # transform tjmat
    for l1 in range(0,lmax+1):
        for l2 in range(0,lmax+1):
            for L in range(0,Lmax+1):
                for M in range(-L,L+1):
                    for m2 in range(-l2,l2+1):
                        for m1 in range(-l1,l1+1):
                            for Mp in range(-L,L+1):
                                tjmat_rot[l1,L,l2,M+L,l1+m1, m2+l2] +=  WDMATS[L][Mp+L,M+L,0] * tjmat[l1,L,l2,L+Mp,l1+m1,l2+m2] 

    return tjmat_rot



def calc_potmatelem_quadpy( l1, m1, l2, m2, rin, scheme, esp_interpolant ):
    """calculate single element of the potential matrix on an interpolated potential"""
    myscheme = quadpy.u3.schemes[scheme]()
    """
    Symbols: 
            theta_phi[0] = theta in [0,pi]
            theta_phi[1] = phi  in [-pi,pi]
            sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) means that we 
                                                                 put the phi angle in range [0,2pi] and 
                                                                 the  theta angle in range [0,pi] as required by 
                                                                 the scipy special funciton sph_harm
    """
    val = myscheme.integrate_spherical(lambda theta_phi: np.conjugate( sph_harm( m1, l1,  theta_phi[1]+np.pi, theta_phi[0] ) ) \
                                                                     * sph_harm( m2, l2, theta_phi[1]+np.pi, theta_phi[0] ) \
                                                                     * POTENTIAL.calc_interp_sph( esp_interpolant, rin, theta_phi[0], theta_phi[1]+np.pi) ) 

    return val

def read_adaptive_quads(params):
    levels = []

    quadfilename = params['job_directory'] + params['file_quad_levels'] 
    fl = open( quadfilename , 'r' )
    for line in fl:
        words   = line.split()
        i       = int(words[0])
        n       = int(words[1])
        xi      = int(words[2])
        scheme  = str(words[3])
        levels.append([i,n,xi,scheme])
    return levels


def read_adaptive_quads_rot(params,irun):
    #for grid calculations on rotated molecules
    levels = []

    quadfilename = params['job_directory'] + params['file_quad_levels'] + "_" + str(irun)

    fl = open( quadfilename , 'r' )
    for line in fl:
        words   = line.split()
        i       = int(words[0])
        n       = int(words[1])
        xi      = int(words[2])
        scheme  = str(words[3])
        levels.append([i,n,xi,scheme])
    return levels

def gen_adaptive_quads(params, esp_interpolant, rgrid):

    sph_quad_list = [] #list of quadrature levels needed to achieve convergence quad_tol of the potential energy matrix elements (global)

    lmax = params['bound_lmax']
    quad_tol = params['sph_quad_tol']

    print("Testing potential energy matrix elements using Lebedev quadratures")

    sphlist = mapping.GEN_SPHLIST(lmax)

    val =  np.zeros( shape = ( len(sphlist)**2 ), dtype=complex)
    val_prev = np.zeros( shape = ( len(sphlist)**2 ), dtype=complex)

    spherical_schemes = []
    for elem in list(quadpy.u3.schemes.keys()):
        if 'lebedev' in elem:
            spherical_schemes.append(elem)

    xi = 0
    for i in range(np.size( rgrid, axis=0 )): 
        for n in range(np.size( rgrid, axis=1 )): 
            rin = rgrid[i,n]
            print("i = " + str(i) + ", n = " + str(n) + ", xi = " + str(xi) + ", r = " + str(rin) )
            if rin <= params['r_cutoff']:
                for scheme in spherical_schemes[3:]: #skip 003a,003b,003c rules
                    k=0
                    for l1,m1 in sphlist:
                        for l2,m2 in sphlist:
                        
                            val[k] = calc_potmatelem_quadpy( l1, m1, l2, m2, rin, scheme, esp_interpolant )
                            print(  '%4d %4d %4d %4d'%(l1,m1,l2,m2) + '%12.6f %12.6fi' % (val[k].real, val[k].imag) + \
                                    '%12.6f %12.6fi' % (val_prev[k].real, val_prev[k].imag) + \
                                    '%12.6f %12.6fi' % (np.abs(val[k].real-val_prev[k].real), np.abs(val[k].imag-val_prev[k].imag)) + \
                                    '%12.6f '%np.abs(val[k]-val_prev[k])  )
                            k += 1

                    diff = np.abs(val - val_prev)

                    if (np.any( diff > quad_tol )):
                        print( str(scheme) + " convergence not reached" ) 
                        for k in range(len(val_prev)):
                            val_prev[k] = val[k]
                    elif ( np.all( diff < quad_tol ) ):     
                        print( str(scheme) + " convergence reached !!!")
                        sph_quad_list.append([ i, n, xi, str(scheme)])
                        xi += 1
                        break

                    #if no convergence reached raise warning
                    if ( scheme == spherical_schemes[len(spherical_schemes)-1] and np.any( diff > quad_tol )):
                        print("WARNING: convergence at tolerance level = " + str(quad_tol) + " not reached for all considered quadrature schemes")

            else:
                scheme = spherical_schemes[lmax+1]
                sph_quad_list.append([ i, n, xi, str(scheme)])
                print("r>r_cutoff: "+str(scheme))
                xi += 1


    print("Converged quadrature levels: ")
    print(sph_quad_list)
    quadfilename = params['job_directory'] + params['file_quad_levels'] 
    fl = open(quadfilename,'w')
    print("saving quadrature levels to file: " + quadfilename )
    for item in sph_quad_list:
        fl.write( str('%4d '%item[0]) + str('%4d '%item[1]) + str('%4d '%item[2]) + str(item[3]) + "\n")

    return sph_quad_list



def gen_adaptive_quads_exact_rot(params , rgrid, mol_xyz, irun ):
    #rotated potential in unrotated basis
    sph_quad_list = [] #list of quadrature levels needed to achieve convergence quad_tol of the potential energy matrix elements (global)

    lmax = params['bound_lmax']
    quad_tol = params['sph_quad_tol']

    print("Adaptive quadratures generator with exact numerical potential")

    sphlist = mapping.GEN_SPHLIST(lmax)

    val =  np.zeros( shape = ( len(sphlist)**2 ), dtype=complex)
    val_prev = np.zeros( shape = ( len(sphlist)**2 ), dtype=complex)

    spherical_schemes = []
    for elem in list(quadpy.u3.schemes.keys()):
        if 'lebedev' in elem:
            spherical_schemes.append(elem)

    xi = 0
    flesp = open("esp_radial",'w')
    esp_int = [] #list of values of integrals for all grid points
    for i in range(np.size( rgrid, axis=0 )): 
        for n in range(np.size( rgrid, axis=1 )): 
            #if i == np.size( rgrid, axis=0 ) - 1 and n == np.size( rgrid, axis=1 ) - 1: break
                        
            rin = rgrid[i,n]
            print("i = " + str(i) + ", n = " + str(n+1) + ", xi = " + str(xi+1) + ", r = " + str(rin) )

            for scheme in spherical_schemes[3:]: #skip 003a,003b,003c rules
                ischeme = 0 #use enumerate()

                #get grid
                Gs = GRID.read_leb_quad(scheme, params['main_dir'] )

                #pull potential at quadrature points
                potfilename = "esp_" + params['molec_name'] + "_"+params['esp_mode'] + "_" + str('%6.4f'%rin) + "_"+scheme + "_"+str(irun)

                if os.path.isfile(params['job_directory']  + "esp/" + str(irun) + "/" + potfilename):
                    print (potfilename + " file exist")


                    filesize = os.path.getsize(params['job_directory']  + "esp/" + str(irun) + "/" + potfilename)

                    if filesize == 0:
                        print("The file is empty: " + str(filesize))
                        os.remove(params['job_directory']  + "esp/"+ str(irun) + "/"  + potfilename)
                        GRID.GEN_XYZ_GRID([Gs], np.array(rin), params['job_directory'] + "esp/"+ str(irun) + "/" )

                        V = GRID.CALC_ESP_PSI4_ROT(params['job_directory'] + "esp/"+ str(irun) + "/" , params, mol_xyz)
                        V = np.asarray(V)
                        fl = open(params['job_directory'] + "esp/"+ str(irun) + "/"  + potfilename,"w")
                        np.savetxt(fl,V,fmt='%10.6f')

                    else:
                        print("The file is not empty: " + str(filesize))
                        fl = open(params['job_directory'] + "esp/" + str(irun) + "/" + potfilename , 'r' )
                        V = []
                        for line in fl:
                            words = line.split()
                            potval = -1.0 * float(words[0])
                            V.append(potval)
                        V = np.asarray(V)
    
                else:
                    print (potfilename + " file does not exist")

                    #generate xyz grid
                    GRID.GEN_XYZ_GRID([Gs], np.array(rin), params['job_directory'] + "esp/"+ str(irun) + "/" )

                    V = GRID.CALC_ESP_PSI4_ROT(params['job_directory'] + "esp/"+ str(irun) + "/" , params, mol_xyz)
                    V = np.asarray(V)

                    #fl = open(params['working_dir'] + "esp/" + potfilename,"w")
                    #np.savetxt(fl, V, fmt='%10.6f')


                for l1,m1 in sphlist:
                    for l2,m2 in sphlist:

                        val[ischeme] = calc_potmatelem_xi( V, Gs, l1, m1, l2, m2 )

                        #val[k] = calc_potmatelem_quadpy( l1, m1, l2, m2, rin, scheme, esp_interpolant )

                        print(  '%4d %4d %4d %4d'%(l1,m1,l2,m2) + '%12.6f' % val[ischeme] + \
                                '%12.6f' % (val_prev[ischeme]) + \
                                '%12.6f '%np.abs(val[ischeme]-val_prev[ischeme])  )
                        ischeme += 1

                diff = np.abs(val - val_prev)

                if (np.any( diff > quad_tol )):
                    print( str(scheme) + " convergence not reached" ) 
                    for ischeme in range(len(val_prev)):
                        val_prev[ischeme] = val[ischeme]
                elif ( np.all( diff < quad_tol ) ):     
                    print( str(scheme) + " convergence reached !!!")
                    
                    if params['integrate_esp'] == True:
                        esp_int.append([xi,rin,val[0]])
                        
                        flesp.write( str('%4d '%xi) + str('%12.6f '%rin) + str('%12.6f '%val[0]) + "\n")

                        
                    sph_quad_list.append([ i, n + 1, xi + 1, str(scheme)]) #new, natural ordering n = 1, 2, 3, ..., N-1, where N-1 is bridge
                    xi += 1
                    break

                #if no convergence reached raise warning
                if ( scheme == spherical_schemes[len(spherical_schemes)-1] and np.any( diff > quad_tol )):
                    print("WARNING: convergence at tolerance level = " + str(quad_tol) + " not reached for all considered quadrature schemes")
                    print( str(scheme) + " convergence reached !!!")

                    sph_quad_list.append([ i, n + 1, xi + 1, str(scheme)])
                    xi += 1

    if params['integrate_esp'] == True:
        print("list of spherical integrals of ESP on radial grid:")
        print(esp_int)
        esp_int = np.asarray(esp_int,dtype=float)
        plt.plot(esp_int[:,1], esp_int[:,2])
        plt.show()
        esp_int_interp  = interpolate.interp1d(esp_int[:,1], esp_int[:,2])
        integral_esp = integrate.quad(lambda x: float(esp_int_interp(x)), 0.05, 200.0)
        exit()

    print("Converged quadrature levels: ")
    print(sph_quad_list)
    quadfilename = params['job_directory'] + params['file_quad_levels'] + "_" + str(irun)
    fl = open(quadfilename,'w')
    print("saving quadrature levels to file: " + quadfilename )
    for item in sph_quad_list:
        fl.write( str('%4d '%item[0]) + str('%4d '%item[1]) + str('%4d '%item[2]) + str(item[3]) + "\n")

    return sph_quad_list



""" ============ PLOTS ============ """
def plot_mat(mat,vmax, show=True,save=False,name="fig",path="./"):
    """ plot 2D array with color-coded magnitude"""
    fig, ax = plt.subplots()
    if sparse.issparse(mat):
        mat = mat.todense()
    im, cbar = heatmap(mat, 0, 0, vmax, ax=ax, cmap="gnuplot", cbarlabel="")
    fig.tight_layout()

    if save == True:
        fig.savefig(path + name + ".pdf" )

    if show == True:
        plt.show()

def heatmap( data, row_labels, col_labels,vmax, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs ):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(np.abs(data), **kwargs)

    # Create colorbar
    im.set_clim(0,vmax ) #np.max(np.abs(data))
    cbar = ax.figure.colorbar(im, ax=ax,    **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    """ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)"""

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    #ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    #ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def plot_wf_rad(rmin,rmax,npoints,coeffs,maparray,rgrid,nlobs,nbins):
    """plot the selected wavefunctions functions"""
    """ Only radial part is plotted"""

    r = np.linspace(rmin, rmax, npoints, endpoint=True, dtype=float)

    x=np.zeros(nlobs)
    w=np.zeros(nlobs)
    x,w=GRID.gauss_lobatto(nlobs,14)
    w=np.array(w)
    x=np.array(x) # convert back to np arrays
    nprint = 2 #how many functions to print

    y = np.zeros((len(r),nprint))

    for l in range(0,nprint): #loop over eigenfunctions
        #counter = 0
        for ipoint in maparray:
            #print(ipoint)
            #print(coeffs[ipoint[4]-1,l])
            y[:,l] +=  coeffs[ipoint[2]-1,l] * chi(ipoint[0],ipoint[1],r,rgrid,w,nlobs,nbins) * SH(ipoint[3], ipoint[4], theta=np.zeros(1), phi=np.zeros(1))
                #counter += 1

    plt.xlabel('r/a.u.')
    plt.ylabel('Radial eigenfunction')
    plt.legend()   
    for n in range(1,nprint):
        plt.plot(r, np.abs(y[:,n])**2)
        
        # plt.plot(r, h_radial(r,1,0))
    plt.show()     

def chi(i,n,r,rgrid,w,nlobs,nbins):
    # r is the argument f(r)
    # rgrid is the radial grid rgrid[i][n]
    # w are the unscaled lobatto weights

    w /=sum(w[:]) #normalization!!!
    val=np.zeros(np.size(r))
    
    if n==0 and i<nbins-1: #bridge functions
        #print("bridge: ", n,i)
        val = ( f(i,nlobs-1,r,rgrid,nlobs,nbins) + f(i+1,0,r,rgrid,nlobs,nbins) ) * np.sqrt( float( w[nlobs-1] ) + float( w[0] ) )**(-1)
        #print(type(val),np.shape(val))
        return val 
    elif n>0 and n<nlobs-1:
        val = f(i,n,r,rgrid,nlobs,nbins) * np.sqrt( float( w[n] ) ) **(-1) 
        #print(type(val),np.shape(val))
        return val
    else:
        return val
         
def f(i,n,r,rgrid,nlobs,nbins): 
    """calculate f_in(r). Input r can be a scalar or a vector (for quadpy quadratures) """
    
    #print("shape of r is", np.shape(r), "and type of r is", type(r))

    if np.isscalar(r):
        prod=1.0
        if  r>= rgrid[i][0] and r <= rgrid[i][nlobs-1]:
            for mu in range(0,nbins):
                if mu !=n:
                    prod*=(r-rgrid[i][mu])/(rgrid[i][n]-rgrid[i][mu])
            return prod
        else:
            return 0.0

    else:
        prod=np.ones(np.size(r), dtype=float)
        for j in range(0,np.size(r)):

            if  r[j] >= rgrid[i,0] and r[j] <= rgrid[i,nlobs-1]:

                for mu in range(0,nbins):
                    if mu !=n:
                        prod[j] *= (r[j]-rgrid[i,mu])/(rgrid[i,n]-rgrid[i,mu])
                    else:
                        prod[j] *= 1.0
            else:
                prod[j] = 0.0
    return prod

def show_Y_lm(l, m):
    """plot the angular basis"""
    theta_1d = np.linspace(0,   np.pi,  2*91) # 
    phi_1d   = np.linspace(0, 2*np.pi, 2*181) # 

    theta_2d, phi_2d = np.meshgrid(theta_1d, phi_1d)
    xyz_2d = np.array([np.sin(theta_2d) * np.sin(phi_2d), np.sin(theta_2d) * np.cos(phi_2d), np.cos(theta_2d)]) #2D grid of cartesian coordinates

    colormap = cm.ScalarMappable( cmap=plt.get_cmap("cool") )
    colormap.set_clim(-.45, .45)
    limit = .5

    print("Y_%i_%i" % (l,m)) # zeigen, dass was passiert
    plt.figure()
    ax = plt.gca(projection = "3d")
    
    plt.title("$Y^{%i}_{%i}$" % (m,l))

    Y_lm = spharm(l,m, theta_2d, phi_2d)
    #Y_lm = self.solidharm(l,m,1,theta_2d,phi_2d)
    print(np.shape(Y_lm))
    r = np.abs(Y_lm.real)*xyz_2d #calculate a point in 3D cartesian space for each value of spherical harmonic at (theta,phi)
    
    ax.plot_surface(r[0], r[1], r[2], facecolors=colormap.to_rgba(Y_lm.real), rstride=1, cstride=1)
    ax.set_xlim(-limit,limit)
    ax.set_ylim(-limit,limit)
    ax.set_zlim(-limit,limit)
    #ax.set_aspect("equal")
    #ax.set_axis_off()
    
            
    plt.show()

def spharm(l,m,theta,phi):
    return sph_harm(m, l, phi, theta)



#if __name__ == "__main__":      


    #params = input.gen_input()

    #BUILD_HMAT0(params)

