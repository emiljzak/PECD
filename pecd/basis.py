#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
#import sys
#print(sys.version)#
import numpy as np
from scipy.special import sph_harm,genlaguerre
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from sympy.physics.quantum.spin import Rotation

from sympy import symbols
from sympy.core import S, Dummy, pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
                                    hermite_poly, jacobi_poly)
from sympy.polys.rootoftools import RootOf
from sympy.core.compatibility import range


class angbas():
    """Class of angular basis functions"""

    def __init__(self):
        """To do: add here attributes of the basis such as lmin,lmax"""
        pass

    def sph_cart(self,r,theta,phi):
        x=r*np.sin(theta)*np.cos(phi)
        y=r*np.sin(theta)*np.sin(phi)
        z=r*np.cos(theta)
        return x,y,z

    def cart_sph(self,x,y,z):
        r=np.sqrt(x**2+y**2+z**2)
        theta=np.arctan(np.sqrt(x**2+y**2)/z)
        phi=np.arctan(y/x)
        return r,theta,phi

    def spharm(self,l,m,theta,phi):
        return sph_harm(m, l, phi, theta)

    def spharmcart(self,l,m,x):
        tol = 1e-8
        #print(np.shape(x[0]))
        if all(np.abs(x[0])>tol) and all(np.abs(x[2])>tol): 
            r=np.sqrt(x[0]**2+x[1]**2+x[2]**2)
            theta=np.arctan(np.sqrt(x[0]**2+x[1]**2)/x[2])
            phi=np.arctan(x[1]/x[0])
        else:
            r = 0
            theta = 0.0
            phi = 0.0
        #print(theta,phi)
        return sph_harm(m, l, phi, theta)

    def spharm_rot(self,l,m,phi,theta,alpha,beta,zgamma):
        #return rotated spherical harmonic in active transformation from MF to LAB frame. 
        Yrot=0.0    
        for mu in range(-l,l+1):
            #print(l,mu,m)
            Yrot+=complex(Rotation.D(l, m, mu , alpha, beta, zgamma).doit().evalf())*self.spharm(l, mu, theta, phi)
        return Yrot

    def solidharm(self,l,m,r,theta,phi):
        if m==0:
            SH=np.sqrt(4.0*np.pi/(2.0*float(l)+1))*(r**l)*np.real(sph_harm(0, l, phi, theta))
        elif m>0:
            SH=((-1.0)**m)*np.sqrt(8.0*np.pi/(2.0*float(l)+1))*(r**l)*np.real(sph_harm(m, l, phi, theta))
        elif m<0:
            SH=((-1.0)**m)*np.sqrt(8.0*np.pi/(2.0*float(l)+1))*(r**l)*np.imag(sph_harm(-m, l, phi, theta))
        return SH

    def show_Y_lm(self,l, m):
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

        Y_lm = self.spharm(l,m, theta_2d, phi_2d)
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

class radbas():
    """Class of radial basis functions"""
    def __init__(self,nlobatto,nbins,binwidth,rshift):
        """
        nlobatto: (int) number of Gauss-Lobatto basis functions per bin
        nbins: (int) nubmer of bins
        """
        self.nlobatto = nlobatto
        self.nbins = nbins
        self.binwidth = binwidth
        self.rshift = rshift #shift of the radial grid away from 0.0 to avoid singularity in the potential (beginning of the radial grid)
        
    def r_grid(self):
        """radial grid of Gauss-Lobatto quadrature points"""        
        #return radial coordinate r_in for given i and n

        x=np.zeros(self.nlobatto)
        w=np.zeros(self.nlobatto)
        x,w=self.gauss_lobatto(self.nlobatto,14)
        w=np.array(w)
        x=np.array(x) # convert back to np arrays
        xgrid=np.zeros((self.nbins,self.nlobatto),dtype=float) # appropriate for row-major language
        
        bingrid = np.zeros(self.nbins)
        bingrid = x * 0.5 * self.binwidth + 0.5 * self.binwidth

        xgrid[:,] = bingrid

        Translvec = np.zeros(self.nbins)
        for i in range(len(Translvec)):
            Translvec[i] = float(i) * self.binwidth + self.rshift

        for ibin in range(self.nbins):    
            xgrid[ibin,:] += Translvec[ibin]

        print('\n'.join([' '.join(["  %12.4f"%item for item in row]) for row in xgrid]))
        
        """for i in range(0,nbins):
                for n in range(0,self.nlobatto):
                    xgrid[i,n]=x[n]*0.5*Rbin+0.5*Rbin+float(i)*Rbin+epsilon
        return xgrid"""
        return xgrid

    def f(self,i,n,r,rgrid): 
        """calculate f_in(r). Input r can be a scalar or a vector (for quadpy quadratures) """
        
        #print("shape of r is", np.shape(r), "and type of r is", type(r))

        if np.isscalar(r):
            prod=1.0
            if  r>= rgrid[i][0] and r <= rgrid[i][self.nlobatto-1]:
                for mu in range(0,self.nlobatto):
                    if mu !=n:
                        prod*=(r-rgrid[i][mu])/(rgrid[i][n]-rgrid[i][mu])
                return prod
            else:
                return 0.0

        else:
            prod=np.ones(np.size(r), dtype=float)
            for j in range(0,np.size(r)):

                if  r[j] >= rgrid[i,0] and r[j] <= rgrid[i,self.nlobatto-1]:

                    for mu in range(0,self.nlobatto):
                        if mu !=n:
                            prod[j] *= (r[j]-rgrid[i,mu])/(rgrid[i,n]-rgrid[i,mu])
                        else:
                            prod[j] *= 1.0
                else:
                    prod[j] = 0.0
        return prod

    def plot_f(self,rmin,rmax,npoints):
        """plot the selected radial basis functions"""
        r = np.linspace(rmin,rmax,npoints,endpoint=True,dtype=float)

        rgrid  = self.r_grid()
        print(np.shape(rgrid))

        y = np.zeros((len(r),self.nlobatto * self.nbins))


        counter = 0
        for i in range(self.nbins):

            for n in range(self.nlobatto):
                y[:,counter] = self.f(i,n,r,rgrid)
                counter += 1
        #print(y)

        figLob = plt.figure()
        plt.xlabel('r/a.u.')
        plt.ylabel('Lobatto basis function')
        plt.legend()   
 
        plt.plot(r, y) 
        plt.show()     

    def fp(self,i,n,k,rgrid):
        "calculate d/dr f_in(r) at r_ik (Gauss-Lobatto grid) " 
        if n!=k:
            fprime=(rgrid[n]-rgrid[k])**(-1)
            prod=1.0e00
            for mu in range(0,self.nlobatto):
                if mu !=k and mu !=n:
                    prod *= (rgrid[k]-rgrid[mu])/(rgrid[n]-rgrid[mu])
            fprime=fprime*prod
        elif n==k:
            fprime=0.0
            for mu in range(0,self.nlobatto):
                    if mu !=n:
                        fprime += (rgrid[n]-rgrid[mu])**(-1)
        return fprime
        
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
        
    def chi(self,i,n,r,rgrid,w):
        # r is the argument f(r)
        # rgrid is the radial grid rgrid[i][n]
        # w are the unscaled lobatto weights

        w /=sum(w[:]) #normalization!!!
        val=np.zeros(np.size(r))
        
        if n==0 and i<self.nbins-1: #bridge functions
            #print("bridge: ", n,i)
            val = ( self.f(i,self.nlobatto-1,r,rgrid) + self.f(i+1,0,r,rgrid) ) * np.sqrt( float( w[self.nlobatto-1] ) + float( w[0] ) )**(-1)
            #print(type(val),np.shape(val))
            return val 
        elif n>0 and n<self.nlobatto-1:
            val = self.f(i,n,r,rgrid) * np.sqrt( float( w[n] ) ) **(-1) 
            #print(type(val),np.shape(val))
            return val
        else:
            return val
            
    def plot_chi(self,rmin,rmax,npoints):
        """plot the selected radial basis functions"""
        r = np.linspace(rmin,rmax,npoints,endpoint=True,dtype=float)

        rgrid  = self.r_grid()
        print(np.shape(rgrid))

        x=np.zeros(self.nlobatto)
        w=np.zeros(self.nlobatto)
        x,w=self.gauss_lobatto(self.nlobatto,14)
        w=np.array(w)
        x=np.array(x) # convert back to np arrays

        y = np.zeros((len(r),self.nlobatto * self.nbins))

        counter = 0
        wcounter = 0
        for i in range(self.nbins):
            for n in range(self.nlobatto):
                y[:,counter] = self.chi(i,n,r,rgrid,w) #* w[wcounter]**(0.5)
                counter += 1
            wcounter +=1

        figLob = plt.figure()
        plt.xlabel('r/a.u.')
        plt.ylabel('Lobatto basis function')
        plt.legend()   
 
        plt.plot(r, y) 
        plt.show()     


    def plot_wf_rad(self,rmin,rmax,npoints,coeffs,maparray,rgrid):
        """plot the selected wavefunctions functions"""
        """ Only radial part is plotted"""

        r = np.linspace(rmin,rmax,npoints,endpoint=True,dtype=float)

        x=np.zeros(self.nlobatto)
        w=np.zeros(self.nlobatto)
        x,w=self.gauss_lobatto(self.nlobatto,14)
        w=np.array(w)
        x=np.array(x) # convert back to np arrays
        nprint = 10 #how many functions to print

        y = np.zeros((len(r),nprint))

        for l in range(0,nprint): #loop over eigenfunctions
            #counter = 0
            for ipoint in maparray:
                #print(ipoint)
                #print(coeffs[ipoint[4]-1,l])
                y[:,l] +=  coeffs[ipoint[4]-1,l] * self.chi(ipoint[2],ipoint[3],r,rgrid,w) 
                    #counter += 1

        plt.xlabel('r/a.u.')
        plt.ylabel('Radial eigenfunction')
        plt.legend()   
        for n in range(1,nprint):
            plt.plot(r, y[:,n]**2)
            
           # plt.plot(r, h_radial(r,1,0))
        plt.show()     

if __name__=="__main__":

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]
    masses=[31.97207070, 1.00782505, 1.00782505]

    """generate mapping function"""
    b = 2 #basis set pruning parameters

    simpleMap = indexmap(b,'simple',3)
    #bas_ind = np.asarray(simpleMap.gen_map())
    bas_ind = simpleMap.gen_map()
    print(type(simpleMap.gen_map()))

    Nbas = np.size(bas_ind , axis =0)
    #print(bas_ind)
    print(type(bas_ind))
    Nquad1D_herm = 8
    Nquad1D_leg = 8
    grid_type = 'dp'

    """Grid types:
                    'dp': direct product Nquad1D_herm x Nquad1D_herm x Nquad1D_leg
                    'ndp_weights': non-direct product with pruning based on product weights (w_tol)
    """

    w_tol =  1e-15 #threshold value for keeping 3D quadrature product-weights

    """generate 3D quadrature grid (for now direct product, only indices)"""
    qgrid = gridmap(Nquad1D_herm, 'dp', 3, w_tol, Nquad1D_herm, Nquad1D_herm, Nquad1D_leg)
    qgrid_ind = np.asarray(qgrid.gen_map())
    Ngrid = np.size(qgrid_ind , axis =0)
    #print(qgrid_ind)

    init(Nbas, Ngrid, Nquad1D_herm, Nquad1D_leg, ref_coords, masses) #hamiltonian class
    hmat(bas_ind,qgrid_ind)

    print("hello")

    #Ylm = angbas()    
    #Ylm.show_Y_lm(l=40,m=40)

    rbas = radbas(nlobatto = 10, nbins = 3, binwidth = 1.0, rshift=0.0)

    #rbas.r_grid()
    """ Test plot the f-radial functions"""
    #rbas.plot_f(rmin = 0.0, rmax = 10.0, npoints = 10000)
    """ Test plot the chi-radial functions"""
    rbas.plot_chi(rmin = 0.0, rmax = 3.0, npoints = 1000)


def matelem_keo( ivec, jvec, psi_i, dpsi_i, psi_j, dpsi_j, x1,x2,x3,  qgrid_ind):
    """
    This routine calculates the matrix element of the kinetic energy operator.

    Input:
    ivec: a vector of shape (3, ) containing i1,i2,i3 (left indices)
    jvec: a vector of shape (3, ) containing j1,j2,j3 (right indices)
    psi_ivec: array (Nquad, 3) of values of the three basis functions (phi_r_i1,phi_r_i2,phi_theta_i3), whose indices correspond to multiindex ivec = (i1,i2,i3).
            The functions are evaluated on respective quadrature grids.
    dpsi_ivec: array (Nquad, 3) of values of the three basis functions derivatives (dphi_r_i1,dphi_r_i2,dphi_theta_i3), whose indices correspond to multiindex ivec = (i1,i2,i3).
            The derivatives are evaluated on respective quadrature grids.
    weights: array (Nquad,) of product quadrature weights weigths[k] = w1[k1] * w2[k2] * w3[k3]
    G: array ((3,3,Nquad, Nquad?)) of values of G-matrix elements on the quadrature grid

    Returns:
    keo_elem: matrix element of the KEO
    """

    keo_elem = np.zeros((1,))
    f_int = np.zeros(np.size(qgrid_ind,axis=0))


    #print(np.shape(keo_jax.Gmat(icoords))) #need to add here full
    keo_jax.init(masses=masses, internal_to_cartesian=internal_to_cartesian)
    for ipoint in range(np.size(qgrid_ind,axis=0)):
        qcoords = [x1[qgrid_ind[ipoint][0]],x2[qgrid_ind[ipoint][1]],x3[qgrid_ind[ipoint][2]]]
        start = time.time()
        G = keo_jax.Gmat(qcoords)
        end = time.time()
        #print("time for keo_jax.Gmat(qcoords) =  ", str(end-start))
        #print(' '.join(["  %15.8f"%item for item in qcoords]))
        #print(dpsi_i[qgrid_ind[ipoint,0], 0 ] * psi_i[qgrid_ind[ipoint,1], 1 ] * psi_i[qgrid_ind[ipoint,2], 2 ] * G[0][0] * dpsi_j[qgrid_ind[ipoint,0], 0 ] * psi_j[qgrid_ind[ipoint,1], 1] * psi_j[qgrid_ind[ipoint,2], 2])
        #print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in G]))

        """start = time.time()
            jax.ops.index_add(f_int,ipoint, dpsi_i[qgrid_ind[ipoint,0], 0 ] * psi_i[qgrid_ind[ipoint,1], 1 ] * psi_i[qgrid_ind[ipoint,2], 2 ] * G[0][0] * dpsi_j[qgrid_ind[ipoint,0], 0 ] * psi_j[qgrid_ind[ipoint,1], 1] * psi_j[qgrid_ind[ipoint,2], 2] \
            + dpsi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ]* G[0][1]* psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + dpsi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ]* G[0][2] * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ]* G[1][0] * dpsi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ]* G[1][1] * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ]* G[1][2] * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ]* G[2][0]* dpsi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ]* G[2][1] * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ]* G[2][2] * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ])

        end = time.time()
        print("time for jax.ops.index_add( =  ", str(end-start))"""


        start = time.time()
        keo_elem += dpsi_i[qgrid_ind[ipoint,0], 0 ] * psi_i[qgrid_ind[ipoint,1], 1 ] * psi_i[qgrid_ind[ipoint,2], 2 ] * dpsi_j[qgrid_ind[ipoint,0], 0 ] * psi_j[qgrid_ind[ipoint,1], 1] * psi_j[qgrid_ind[ipoint,2], 2] \
            + dpsi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + dpsi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * dpsi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ]* dpsi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ] * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ] * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ]
        end = time.time()
        print("time for jax.ops.index_add =  ", str(end-start))

    #jax.ops.index_update(f,ipoint,f_int)
    #print(type(f))
    #keo_elem = np.sum(f)
    #print(f_int)
    print("we are returning value")

    return keo_elem