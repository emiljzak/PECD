#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2022 Emil Zak <emil.zak@cfel.de>
#
from logging import raiseExceptions
import numpy as np

import itertools

#from numba import jit, prange
#jitcache = False


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
        #generate mapping for general FEMLIST
        if self.map_type == 'DVR':
            maparray, Nbas = self.map_dvr_femlist_nat()
        elif self.map_type == 'SPECT':
            maparray, Nbas = self.map_spect_femlist()

        #fl = open(working_dir + 'map.dat','w')
        #for elem in maparray:   
        #    fl.write("%5d"%elem[0]+"  %5d"%elem[1]+ "  %5d"%elem[2]+ "  %5d"%elem[3]+  " %5d"%elem[4]+" %5d"%elem[5]+"\n")
        #fl.close()
        
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

        imap        = 0
        xi          = 0
        maparray    = []
        ibincount   = -1
        nbins = 0

        for elem in self.femlist:
            nbins += elem[0]
        print("total number of bins = " + str(nbins))

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


        Nbas = imap
        return maparray, Nbas


    def map_dvr_femlist_nat(self):
        """ 
        Generates an index map with grid points as major dimension and bridge points/functions placed as last in the bin.

        Returns: tuple
            maparray: list
                [[ibin,n,ipoint,l,m,id],...]
            Nbas: int
                Number of basis functions

        Note: the convention adopted to generate the index map assumes that all radial indices start with 0.
                This python-compatible convention helps to clear out any confusions.
                The resulting matrix structure is close to block-diagonal, with single-point sized bridges between bins, as shown in the image below:
        .. image:: /_images/ham_nat.png
            :width: 500
        Examples:
                ibin  n     ipoint l    m   id

        """

        #natural order of grid points and basis functions, including bridges
        imap = 0
        xi = 0
        maparray = []
        ibincount = -1

        nbins = 0
        for elem in self.femlist:
            nbins += elem[0]
        print("total number of bins = " + str(nbins))

        for elem in self.femlist:
            for i in range(elem[0]):
                ibincount +=1
                for n in range(1,elem[1]):
                    if ibincount == nbins-1 and n == elem[1]-1:
                        continue     
                    else:
                        xi += 1
                        for l in range(0,self.lmax+1):
                            for m in range(-l,l+1):

                                imap += 1
                                #print(ibincount,n,xi,l,m,imap)
                                maparray.append([ibincount,n,xi,l,m,imap])

        Nbas = imap
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

        return maparray, imap




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


    def GEN_KLIST(self,maparray, Nbas, map_type):
        #create a list of indices for matrix elements of the KEO
        klist = []

        if map_type == 'DVR':
            for p1 in range(Nbas):
                for p2 in range(p1, Nbas):
                    if maparray[p1][3] == maparray[p2][3] and maparray[p1][4] == maparray[p2][4]:
                        if maparray[p1][0] == maparray[p2][0] or maparray[p1][0] == maparray[p2][0] - 1 or maparray[p1][0] == maparray[p2][0] + 1: 
                            klist.append([ maparray[p1][0], maparray[p1][1], maparray[p2][0], \
                            maparray[p2][1], maparray[p2][3], p1, p2 ])


        return klist

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

    def __init__(self,N_euler,N_batches,grid_type):

        self.N_euler    = N_euler
        self.N_batches  = N_batches
        self.grid_type  = grid_type


    def read_euler_grid(self):   

        with open( "grid_euler.dat" , 'r') as eulerfile:   
            grid_euler = np.loadtxt(eulerfile)

        grid_euler = grid_euler.reshape(-1,3)     

        N_Euler = grid_euler.shape[0]

        N_per_batch = int(N_Euler/self.Nbatches)

        return grid_euler, N_Euler, N_per_batch


    def gen_euler_grid_2D(self):
        """ Cartesian product of 1D grids of Euler angles"""
        alpha_1d        = list(np.linspace(0, 2*np.pi,  num=1, endpoint=False))
        beta_1d         = list(np.linspace(0, np.pi,    num=self.N_euler, endpoint=True))
        gamma_1d        = list(np.linspace(0, 2*np.pi,  num=self.N_euler, endpoint=False))
        euler_grid_3d   = np.array(list(itertools.product(*[alpha_1d, beta_1d, gamma_1d]))) #cartesian product of [alpha,beta,gamma]

        n_euler_3d      = euler_grid_3d.shape[0]
        print("\nTotal number of 2D-Euler grid points: ", n_euler_3d , " and the shape of the 3D grid array is:    ", euler_grid_3d.shape)
        #print(euler_grid_3d)
        return euler_grid_3d, n_euler_3d

    def gen_euler_grid(self):
        """ Cartesian product of 1D grids of Euler angles"""
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
        self.nlobs = nlobs
        self.nbins = nbins
        self.binwidth = binwidth
        self.rshift = rshift


    def gen_grid(self):
        """ Generates radial grid based on Gauss-Lobatto quadrature points
            Three grids are returned: quadrature grid, primitive grid, coupled grid used in final calculation.

        Returns: tuple
        x: numpy 1D array (float)
            Gauss-lobatto quadrature grid

        rgrid_prim: numpy 1D array (float)
            Primitive radial grid containing all grid points, plus point values at bin boundries are duplicated i.e. first point in the bin has equal value to last point from previous bin.
        
        rgrid: numpy 1D array (float)
            Coupled radial grid with boundary points excluded and no duplicate points at bin boundaries.

        ToDo: this function must be generalized to account for FEMLIST. Presently only constant bin size is possible."""

        nlobs = self.nlobs
        nbins = self.nbins
        binwidth = self.binwidth
        rshift = self.rshift

        x       = np.zeros(nlobs, dtype = float)
        w       = np.zeros(nlobs, dtype = float)
        x, w    = self.gauss_lobatto(nlobs,14)
        w       = np.array(w)
        x       = np.array(x)

        rgrid   = np.zeros( (nbins, nlobs-1), dtype = float) 
        rgrid_prim = 0

        bingrid     = np.zeros(nbins)
        bingrid     = x[1:] * 0.5 * binwidth + 0.5 * binwidth
        rgrid[:,]   = bingrid
        Translvec   = np.zeros(nbins)

        for i in range(len(Translvec)):
            Translvec[i] = float(i) * binwidth + rshift

        for ibin in range(nbins):    
            rgrid[ibin,:] += Translvec[ibin]

        print('\n'.join([' '.join(["  %12.4f"%item for item in row]) for row in xgrid]))

        return x, rgrid_prim, rgrid 

    def r_grid_prim(self,nlobatto,nbins,binwidth,rshift):
        """radial grid of Gauss-Lobatto quadrature points"""        
        #return radial coordinate r_in for given i and n
        #we double count joining points. It means we work in primitive basis/grid

        """Note: this function must be generalized to account for FEMLIST"""

        x = np.zeros(nlobatto)
        w = np.zeros(nlobatto)
        x, w = gauss_lobatto(nlobatto,14)
        w = np.array(w)
        x = np.array(x) # convert back to np arrays
        xgrid = np.zeros( (nbins,nlobatto), dtype=float) 
        
        bingrid = np.zeros(nbins)
        bingrid = x * 0.5 * binwidth + 0.5 * binwidth
        xgrid[:,] = bingrid

        Translvec = np.zeros(nbins)

        for i in range(len(Translvec)):
            Translvec[i] = float(i) * binwidth + rshift

        for ibin in range(nbins):    
            xgrid[ibin,:] += Translvec[ibin]

        #print('\n'.join([' '.join(["  %12.4f"%item for item in row]) for row in xgrid]))

        return xgrid, nlobatto * nbins
