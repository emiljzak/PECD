#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
#import sys
#print(sys.version)#
import numpy as np
from scipy.special import sph_harm
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
        
        bingrid = np.zeros(self.nlobatto)
        bingrid = x*0.5*self.binwidth+0.5*self.binwidth

        xgrid[:,] = bingrid

        Translvec = np.zeros(self.nlobatto)
        for i in range(len(Translvec)):
            Translvec[i] = float(i) * self.binwidth 

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
            fprime=(rgrid[n]-rgrid[k])
            prod=1.0e00
            for mu in range(0,self.nlobatto):
                if mu !=k and mu !=n:
                    prod=prod*(rgrid[k]-rgrid[mu])/(rgrid[n]-rgrid[mu])
            fprime=fprime*prod
        elif n==k:
            fprime=1.0
            summ=0.0e00
            for mu in range(0,self.nlobatto):
                    if mu !=n:
                        summ=summ+(rgrid[n]-rgrid[mu])**(-1)
            fprime=fprime*summ
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

        w *= (0.5 * self.binwidth)#*sum(w[:])
        val=np.zeros(np.size(r))
        
        if n==0 and i<self.nbins-1:
            #print("bridge: ", n,i)
            val = ( self.f(i,self.nlobatto-1,r,rgrid) + self.f(i+1,0,r,rgrid) ) * np.sqrt( float( w[self.nlobatto-1] ) + float( 2.0 *w[0] ) )**(-1)
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
        for i in range(self.nbins):

            for n in range(self.nlobatto):
                y[:,counter] = self.chi(i,n,r,rgrid,w)
                counter += 1

        figLob = plt.figure()
        plt.xlabel('r/a.u.')
        plt.ylabel('Lobatto basis function')
        plt.legend()   
 
        plt.plot(r, y) 
        plt.show()     


if __name__ == "__main__":      
    print("hello")

    #Ylm = angbas()    
    #Ylm.show_Y_lm(l=40,m=40)

    rbas = radbas(nlobatto = 10, nbins = 3, binwidth = 1.0, rshift=0.0)

    #rbas.r_grid()
    """ Test plot the f-radial functions"""
    #rbas.plot_f(rmin = 0.0, rmax = 10.0, npoints = 10000)
    """ Test plot the chi-radial functions"""
    rbas.plot_chi(rmin = 0.0, rmax = 3.0, npoints = 1000)