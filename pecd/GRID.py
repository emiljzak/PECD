#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
import os

import psi4

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

def read_leb_quad(scheme):
    sphgrid = []
    #print("reading Lebedev grid from file:" + "/lebedev_grids/"+str(scheme)+".txt")
    fl = open( "/Users/zakemil/Nextcloud/projects/PECD/pecd/" + "lebedev_grids/" + str(scheme) + ".txt", 'r' )
    for line in fl:
        words   = line.split()
        phi     = float(words[0])
        theta   = float(words[1])
        w       = float(words[2])
        sphgrid.append([theta,phi,w])

    sphgrid = np.asarray(sphgrid)
    sphgrid[:,0:2] = np.pi * sphgrid[:,0:2] / 180.0 #to radians

    """
    xx = np.sin(sphgrid[:,1])*np.cos(sphgrid[:,0])
    yy = np.sin(sphgrid[:,1])*np.sin(sphgrid[:,0])
    zz = np.cos(sphgrid[:,1])
    #Set colours and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx,yy,zz,color="k",s=20)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    plt.tight_layout()
    #plt.show()
    """
    return sphgrid

def GEN_GRID(sph_quad_list):
    Gs = []
    for elem in sph_quad_list:
        gs = read_leb_quad(str(elem[3]))
        Gs.append( gs )

    return Gs


def gauss_lobatto(n, n_digits):
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
    

def r_grid(nlobatto,nbins,binwidth,rshift):
    """radial grid of Gauss-Lobatto quadrature points"""        
    #return radial coordinate r_in for given i and n

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

    print('\n'.join([' '.join(["  %12.4f"%item for item in row]) for row in xgrid]))
    
    return xgrid, nlobatto * nbins

def sph2cart(r,theta,phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x,y,z

def GEN_XYZ_GRID(Gs,Gr,working_dir):
    grid = []
    r_array = Gr.flatten()

    gridfile = open(working_dir + "grid.dat", 'w')

    for k in range(len(r_array)):
        #print(Gs[k].shape[0])
        for s in range(Gs[k].shape[0]):

            theta   = Gs[k][s,0]
            phi     = Gs[k][s,1]
            r       = r_array[k]
            x,y,z   = sph2cart(r,theta,phi)

            gridfile.write( " %12.6f"%x +  " %12.6f"%y + "  %12.6f"%z + "\n")
            grid.append([x,y,z])

    return grid

def CALC_ESP_PSI4(dir):
    os.chdir(dir)
    psi4.core.be_quiet()
    properties_origin=["COM"]
    #psi4.core.set_output_file(dir, False)
    h2o = psi4.geometry("""
    1 2
    noreorient
    O            0.0     0.0    0.000000000000
    H            0.757    0.586     0.000000000000
    H            -0.757    0.586     0.000000000000
    """)
    psi4.set_options({'basis': '6-31G**','e_convergence': 1.e-5,'reference': 'uhf'})
    E, wfn = psi4.prop('scf', properties=["GRID_ESP"], return_wfn=True)
    Vvals = wfn.oeprop.Vvals()
    os.chdir("../")
    return Vvals
