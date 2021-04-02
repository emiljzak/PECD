#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np


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
