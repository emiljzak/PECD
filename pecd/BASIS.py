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



