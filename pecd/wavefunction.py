#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2022 Emil Zak <emil.zak@cfel.de>
#
from logging import raiseExceptions
import numpy as np

import itertools



#For Gauss-Lobatto
from sympy import symbols
from sympy.core import S, Dummy, pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
                                    hermite_poly, jacobi_poly)
from sympy.polys.rootoftools import RootOf


#from numba import jit, prange
#jitcache = False

class Psi():

    def __init__(self, params):

        self.params  = params


class Map():
    """Map class keeps methods for generating and manipulating index mapping for the wavefunction and matrices.

    Note:
        The characteristics of the photo-electron propagation using the FEM-DVR method suggests
        that a DVR-type mapping is suitable and natural.

    Attributes:
        femlist (list): keeps sizes of radial bins and the number of points/functions in each bin.
                        Format: [ [nbins,nlobs,Rbin], ... ].

        map_type (str): type of mapping function ('DVR' or 'SPECT')

        job_dir (str): path to current working directory

    Parameters:
        lmax (int): maximum value of the l quantum number

    """
    def __init__(self, femlist, map_type, job_dir, lmax = 0):

        self.femlist    = femlist
        self.map_type   = map_type
        self.job_dir    = job_dir
        self.lmax       = lmax 


    def save_map(self,maparray,file):
        fl = open(self.job_dir+file,'w')
        for elem in maparray:   
            fl.write(" ".join('{:5d}'.format(elem[i]) for i in range(0,6)) + "\n")
        fl.close()


    def genmap_femlist(self):
        """ 
        Driver routine to call map generators

        Returns: tuple
            maparray: list
                [[ibin,n,ipoint,l,m,id],...]
            Nbas: int
                Number of basis functions
        """

        if self.map_type == 'DVR':
            maparray, Nbas = self.map_dvr_femlist_nat()
        elif self.map_type == 'SPECT':
            maparray, Nbas = self.map_spect_femlist()

        return maparray, Nbas



    def map_dvr_femlist(self):
        """ 
        Generates an index map with grid points as major dimension and bridge points/functions placed as first in the bin.

        Returns: tuple
            maparray: list
                [[ibin,n,ipoint,l,m,id],...]
            Nbas: int
                Number of basis functions
        """

        imap        = -1
        xi          = -1
        maparray    = []
        ibincount   = -1
        nbins = 0

        for elem in self.femlist:
            nbins += elem[0]


        for elem in self.femlist:
            for i in range(elem[0]):
                ibincount +=1
                for n in range(elem[1]):
                    if n == elem[1]-1:
                        continue   
                    elif n == 0 and ibincount == nbins-1:
                        continue     
                    else:
                        xi += 1
                        for l in range(0,self.lmax+1):
                            for m in range(-l,l+1):

                                imap += 1
                                #print(ibincount,n,xi,l,m,imap)
                                maparray.append([ibincount,n,xi,l,m,imap])


        Nbas = imap + 1
        return maparray, Nbas


    def map_dvr_femlist_nat(self):
        """ 
        Generates an index map for coupled basis with grid points as major dimension and bridge points/functions placed as last in the bin.

        Returns: tuple
            maparray: list
                [[ibin,n,ipoint,l,m,id],...]
            Nbas: int
                Number of basis functions

        Note:   the convention adopted to generate the index map assumes that all radial indices start with 0. This python-compatible convention helps to clear out any confusions. The resulting matrix structure is close to block-diagonal, with single-point sized bridges between bins, as shown in the image below:
        
        .. image:: /_images/ham_nat.png
            :width: 500
        Examples:
                ibin  n     ipoint l    m   id 

        Status: tested vs analytic formula. OK.
        """

        imap = -1
        xi = -1
        maparray = []
        ibincount = -1

        nbins_tot = 0
        for elem in self.femlist:
            nbins_tot += elem[0]
       
        for elem in self.femlist:
            for i in range(elem[0]):
                ibincount +=1
                for n in range(0,elem[1]-1):
                    if ibincount == nbins_tot-1 and n == elem[1]-2:
                        continue     
                    else:
                        xi += 1
                        for l in range(0,self.lmax+1):
                            for m in range(-l,l+1):

                                imap += 1
                                #print(ibincount,n,xi,l,m,imap)
                                maparray.append([ibincount,n,xi,l,m,imap])

        Nbas = imap + 1

        return maparray, Nbas


    def map_spect_femlist(self,femlist,lmax):
        ##### NEEDS VERIFICATION #########
        imap = 0
        xi = 0
        maparray = []
        ibincount = -1

        nbins = 0
        for elem in femlist:
            nbins += elem[0]
        print("total number of bins = " + str(nbins))
        
        for l in range(0,lmax+1):
            for m in range(-l,l+1):
                xi = 0
                ibinscound = -1
                for elem in femlist:
                    for i in range(elem[0]):
                        ibincount +=1
                        for n in range(elem[1]):
                            if n == elem[1]-1:
                                continue   
                            elif n == 0 and ibincount == nbins-1:
                                continue     
                            else:
                                xi += 1
                                imap += 1
                                #print(l,m,ibincount,n,xi,imap)
                                maparray.append([l,m,ibincount,n,xi,imap])

        return maparray, imap +1


    def GEN_SPHLIST(self,lmax):
        #create list of basis set indices for adaptive quadratures
        sphlist = []
        if lmax == 0:
            raise ValueError("lmax = 0 makes no sense in the generation of adaptive quadratures")

        for l in range(lmax-1,lmax+1):
            for m in range(l-1,l+1):
                sphlist.append([l,m])
        return sphlist


    #@jit( nopython=True, parallel=False, cache = jitcache, fastmath=False) 
    def GEN_VLIST(self,maparray, Nbas, map_type):
        #create a list of indices for matrix elements of the potential
        vlist = []

        #needs speed-up: convert maparray to numpy array and vectorize, add numpy

        if map_type == 'DVR':
            for p1 in range(Nbas):
                for p2 in range(p1, Nbas):
                    if maparray[p1][2] == maparray[p2][2]: 
                        vlist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], \
                            maparray[p2][3], maparray[p2][4], p1, p2  ])

        elif map_type == 'SPECT':
            for p1 in range(Nbas):
                for p2 in range(p1, Nbas):
                    if maparray[p1][2] == maparray[p2][2]:
                        vlist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], maparray[p2][3], maparray[p2][4] ])
        return vlist

    def calc_p2(l,m,xi,lmax):
        return (xi-1)*(lmax+1)**2 + l*(l+1) + m


    def GEN_DIPLIST_opt1(self,maparray, Nbas, lmax, map_type ):
        """
        create a list of indices for matrix elements of the dipole interaction matrix. In the future: vectorize by copying the core array. 
        """
        sigma = 1 #we always calculate h_ll'mm',+1 element and generate h_ll'mm',-1 with the hermitian conjugate (regardless of the light's helicity) 
        diplist = []
        if map_type == 'DVR':
            for p1 in range(Nbas):

                xi = maparray[p1][2] 
                l1 = maparray[p1][3]
                m1 = maparray[p1][4]

                if l1+1 <= lmax:
                    p2 = calc_p2(l1+1,m1-sigma,xi,lmax)
                    diplist.append([ xi, l1, m1, l1+1, m1-sigma, p1, p2 ])
                if l1-1 >= 0:
                    p2 = calc_p2(l1-1,m1-sigma,xi,lmax)
                    diplist.append([ xi, l1, m1, l1-1, m1-sigma, p1, p2 ])
        else:
            ValueError("Incorrect map type")

        return diplist


    def GEN_DIPLIST(self,maparray, Nbas, map_type, sigma ):
        """
        Old O(n**2) implementation.
        create a list of indices for matrix elements of the dipole interaction matrix. 
        Generates full square matrix for sigma = -1 or 0. 
        sigma = +1 can be generated using symmetries.
        
        """
        
        diplist = []
        if map_type == 'DVR':

            #set up indices for block-listing

            if sigma == 0:

                for p1 in range(Nbas):
                    for p2 in range(Nbas):
                        if maparray[p1][2] == maparray[p2][2]: 

                            if maparray[p1][3] == maparray[p2][3] - 1 or maparray[p1][3] == maparray[p2][3] + 1:
                                if maparray[p1][4] == maparray[p2][4]: 
                                    diplist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], \
                                    maparray[p2][3], maparray[p2][4], p1, p2 ])

            elif sigma == +1:

                #for xi in range(Nr-1):
                #    for i1 in range(block_size):
                for p1 in range(Nbas):          
                    for p2 in range(Nbas):
                        if maparray[p1][2] == maparray[p2][2]: 

                            if maparray[p1][3] == maparray[p2][3] - 1 or maparray[p1][3] == maparray[p2][3] + 1:
                                if maparray[p2][4] == maparray[p1][4] + 1: 
                                    diplist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], \
                                    maparray[p2][3], maparray[p2][4], p1, p2 ])

            elif sigma == -1:
                for p1 in range(Nbas):
                    for p2 in range(Nbas):
                        if maparray[p1][2] == maparray[p2][2]: 

                            if maparray[p1][3] == maparray[p2][3] - 1 or maparray[p1][3] == maparray[p2][3] + 1:
                                if maparray[p2][4] == maparray[p1][4] - 1: 
                                    diplist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], \
                                    maparray[p2][3], maparray[p2][4], p1, p2 ])
        else:
            ValueError("Incorrect map type")



        return diplist

class GridEuler():
    """GridEuler class keeps methods for generating and manipulating the grid of molecular orientations

    Attributes:
        N_euler (int): number of molecular molecular orientations per dimension

        N_batches (int): number of batches into which the full run is divided

        grid_type (str): 2D or 3D grid type. For 2D the first Euler angle (alpha) is fixed at 0.0.


    """



    def __init__(self,N_euler,N_batches,grid_type):

        self.N_euler    = N_euler
        self.N_batches  = N_batches
        self.grid_type  = grid_type


    def read_euler_grid(self):   

        with open( "grid_euler.dat" , 'r') as eulerfile:   
            grid_euler = np.loadtxt(eulerfile)

        grid_euler = grid_euler.reshape(-1,3)     

        N_Euler = grid_euler.shape[0]

        N_per_batch = int(N_Euler/self.N_batches)

        return grid_euler, N_Euler, N_per_batch


    def gen_euler_grid_2D(self):
        """
        Generates the 2D Cartesian product of 1D grids of Euler angles. 

        Returns: tuple
                euler_grid: numpy 2D array
                    grid of euler angles [alpha=0.0,beta,gamma]
                n_euler: int
                    the total number of grid points

        Status: tested.

        """
        alpha_1d        = list(np.linspace(0, 2*np.pi,  num=1, endpoint=False))
        beta_1d         = list(np.linspace(0, np.pi,    num=self.N_euler, endpoint=True))
        gamma_1d        = list(np.linspace(0, 2*np.pi,  num=self.N_euler, endpoint=False))
        euler_grid   = np.array(list(itertools.product(*[alpha_1d, beta_1d, gamma_1d]))) #cartesian product of [alpha,beta,gamma]

        n_euler      = euler_grid.shape[0]
        print("\nTotal number of 2D-Euler grid points: ", n_euler , " and the shape of the 3D grid array is:    ", euler_grid.shape)
        #print(euler_grid_3d)
        return euler_grid, n_euler

    def gen_euler_grid(self):
        """
        Generates the 3D Cartesian product of 1D grids of Euler angles. 

        Returns: tuple
                euler_grid: numpy 3D array
                    grid of euler angles [alpha,beta,gamma]
                n_euler: int
                    the total number of grid points

        Status: tested
        """
        alpha_1d        = list(np.linspace(0, 2*np.pi,  num=self.N_euler, endpoint=False))
        beta_1d         = list(np.linspace(0, np.pi,    num=self.N_euler, endpoint=True))
        gamma_1d        = list(np.linspace(0, 2*np.pi,  num=self.N_euler, endpoint=False))
        euler_grid_3d   = np.array(list(itertools.product(*[alpha_1d, beta_1d, gamma_1d]))) #cartesian product of [alpha,beta,gamma]

        n_euler_3d      = euler_grid_3d.shape[0]
        print("\nTotal number of 3D-Euler grid points: ", n_euler_3d , " and the shape of the 3D grid array is:    ", euler_grid_3d.shape)
        #print(euler_grid_3d)
        return euler_grid_3d, n_euler_3d

    def save_euler_grid(self, grid_euler, path):   
        with open( path + "grid_euler.dat" , 'w') as eulerfile:   
            np.savetxt(eulerfile, grid_euler, fmt = '%15.4f')


class GridRad():
    """Class contains methods related to the radial grid.

    Attrs:
        nlobs : int
            number of Gauss-Lobatto functions per bin
        nbins : int
            number of bins
        binwidth : float
            width (a.u.) of the bin
        rshift : float
            shift of the grid (a.u.)
    """

    def __init__(self,nlobs,nbins,binwidth,rshift):
        self.nlobs      = nlobs
        self.nbins      = nbins
        self.binwidth   = binwidth
        self.rshift     = rshift


    def gen_grid(self):
        """ Generates radial grid based on Gauss-Lobatto quadrature points
            Three grids are returned: quadrature grid, primitive grid, coupled grid used in final calculation.

        Returns: tuple
        x: numpy 1D array (float, size = nlobs)
            Gauss-lobatto quadrature grid

        w: numpy 1D array (float, size = nlobs)
            Gauss-lobatto quadrature weights

        rgrid_prim: numpy 1D array (float, size = nlobs * nbins)
            Primitive radial grid containing all grid points, plus point values at bin boundries are duplicated i.e. first point in the bin has equal value to last point from previous bin.
        
        rgrid: numpy 1D array (float, size = Nr = (nlobs - 1)*nbins - 1)
            Coupled radial grid with boundary points excluded and no duplicate points at bin boundaries.

        Nr: int
            number of coupled grid points

        Todo: this function must be generalized to account for FEMLIST. Presently only constant bin size is possible.

        Status: tested
        """

        nlobs           = self.nlobs
        nbins           = self.nbins
        binwidth        = self.binwidth
        rshift          = self.rshift

        #number of coupled radial grid points
        Ngridcoupled    = nbins*(nlobs-1)-1

        #number of primitive radial grid points
        Ngridprim       = nbins * nlobs

        x               = np.zeros(nlobs, dtype = float)
        w               = np.zeros(nlobs, dtype = float)
        xx, ww            = self.gauss_lobatto(nlobs,14)

        for k in range(nlobs):
            x[k] = float(xx[k]) 
            w[k] = float(ww[k]) 
        
        #w               = np.asarray(w,dtype=float)
       # x               = np.asarray(x,dtype=float)

        xgrid_prim      = np.zeros((nbins, nlobs), dtype = float) 
        xgrid_coupled   = np.zeros((nbins, nlobs-1), dtype = float) 
        rgrid           = np.zeros(Ngridcoupled, dtype = float)
        rgrid_prim      = np.zeros(Ngridprim, dtype = float)

        #broadcast GL quadrature into full grid
        xgrid_coupled[:,]   = x[1:] * 0.5 * binwidth + 0.5 * binwidth
        xgrid_prim[:,]      = x * 0.5 * binwidth + 0.5 * binwidth
        
        Translvec   = np.zeros(nbins)

        for i in range(len(Translvec)):
            Translvec[i] = float(i) * binwidth + rshift

        for ibin in range(nbins):
            xgrid_prim[ibin,:]      += Translvec[ibin]
            xgrid_coupled[ibin,:]   += Translvec[ibin]

        #remove last point from coupled grid
        rgrid       = xgrid_coupled.flatten()[:Ngridcoupled]
        rgrid_prim  = xgrid_prim.flatten()[:Ngridprim]

        #print("Coupled radial grid:")
        #print('\n'.join([' '.join(["  %12.4f"%item for item in row]) for row in xgrid_coupled]))
        #print("\n")
        Nr = Ngridcoupled
        return x, w, rgrid_prim, rgrid, Nr

  
    def gauss_lobatto(self,n, n_digits):
        """
        Computes the Gauss-Lobatto quadrature [1]_ points and weights.

        The Gauss-Lobatto quadrature approximates the integral:

        .. math::
            \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

        The nodes `x_i` of an order `n` quadrature rule are the roots of `P'_(n-1)`
        and the weights `w_i` are given by:

        .. math::
            &w_i = \frac{2}{n(n-1) \left[P_{n-1}(x_i)\right]^2},\quad x\neq\pm 1\\
            &w_i = \frac{2}{n(n-1)},\quad x=\pm 1

        Parameters
        ==========

        n : the order of quadrature

        n_digits : number of significant digits of the points and weights to return

        Returns
        =======

        (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
                The points `x_i` and weights `w_i` are returned as ``(x, w)``
                tuple of lists.

        Examples
        ========

        >>> from sympy.integrals.quadrature import gauss_lobatto
        >>> x, w = gauss_lobatto(3, 5)
        >>> x
        [-1, 0, 1]
        >>> w
        [0.33333, 1.3333, 0.33333]
        >>> x, w = gauss_lobatto(4, 5)
        >>> x
        [-1, -0.44721, 0.44721, 1]
        >>> w
        [0.16667, 0.83333, 0.83333, 0.16667]

        See Also
        ========

        gauss_legendre,gauss_laguerre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules
        .. [2] http://people.math.sfu.ca/~cbm/aands/page_888.htm
        """
        x = Dummy("x")
        p = legendre_poly(n-1, x, polys=True)
        pd = p.diff(x)
        xi = []
        w = []
        for r in pd.real_roots():
            if isinstance(r, RootOf):
                r = r.eval_rational(S(1)/10**(n_digits+2))
            xi.append(r.n(n_digits))
            w.append((2/(n*(n-1) * p.subs(x, r)**2)).n(n_digits))

        xi.insert(0, -1)
        xi.append(1)
        w.insert(0, (S(2)/(n*(n-1))).n(n_digits))
        w.append((S(2)/(n*(n-1))).n(n_digits))
        return xi, w