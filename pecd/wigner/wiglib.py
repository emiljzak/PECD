"""Wigner D-functions
"""
import numpy as np
import sys
import os
from ctypes import c_double, c_int, POINTER


# load Fortran library wiglib
_wiglib_path = os.path.join(os.path.dirname(__file__), 'wiglib')
_wiglib = np.ctypeslib.load_library('wiglib', _wiglib_path)


def DJmk(J, grid):
    """Computes Wigner D-functions D_{m,k}^{(J)} on a grid of Euler angles, for k,m=-J..J.

    Parameters
    ----------
    J : int
       Value J quantum number.
    grid : array (3,npoints)
        Numpy array containing grid of Euler angles, grid[:3,ipoint] = (phi,theta,chi),
        where "phi" and "chi" are Euler angles associated with the "m" and "k" quantum numbers,
        respectively, and "ipoint" in range(npoints)
    Returns
    -------
    wig : array (npoints, 2*J+1, 2*J+1)
        Contains values of D-functions on grid, D_{m,k}^{(J)} = wig[ipoint,m+J,k+J]
    """

    npoints_3d = grid.shape[1]
    grid_3d = np.asfortranarray(grid)

    Dr = np.asfortranarray(np.zeros((npoints_3d,2*J+1,2*J+1), dtype=np.float64)) # (ipoint,m,k)
    Di = np.asfortranarray(np.zeros((npoints_3d,2*J+1,2*J+1), dtype=np.float64))

    npoints_c = c_int(npoints_3d)
    J_c = c_int(J)

    _wiglib.DJmk.argtypes = [ c_int, c_int, \
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F'), \
        np.ctypeslib.ndpointer(np.float64, ndim=3, flags='F'), \
        np.ctypeslib.ndpointer(np.float64, ndim=3, flags='F') ]

    _wiglib.DJmk.restype = None

    _wiglib.DJmk( J_c, npoints_c, grid_3d, Dr, Di )

    wig = Dr.reshape((npoints_3d,2*J+1,2*J+1)) + Di.reshape((npoints_3d,2*J+1,2*J+1))*1j # (ipoint,m,k)

    return wig


def DJ_m_k(J, m, grid):
    """Computes Wigner D-functions D_{m,k}^{(J)} on a grid of Euler angles, for k=-J..J and selected m.

    Parameters
    ----------
    J : int
       Value of J quantum number.
    m : int
       Value of m quantum number.
    grid : array (3,npoints)
        Numpy array containing grid of Euler angles, grid[:3,ipoint] = (phi,theta,chi),
        where "phi" and "chi" are Euler angles associated with the "m" and "k" quantum numbers,
        respectively, and "ipoint" in range(npoints)
    Returns
    -------
    wig : array (npoints, 2*J+1)
        Contains values of D-functions on grid, D_{m,k}^{(J)} = wig[ipoint,k+J]
    """

    npoints_3d = grid.shape[1]
    grid_3d = np.asfortranarray(grid)

    Dr = np.asfortranarray(np.zeros((npoints_3d,2*J+1), dtype=np.float64)) # (ipoint,k)
    Di = np.asfortranarray(np.zeros((npoints_3d,2*J+1), dtype=np.float64))

    npoints_c = c_int(npoints_3d)
    J_c = c_int(J)
    m_c = c_int(m)

    _wiglib.DJ_m_k.argtypes = [ c_int, c_int, c_int, \
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F'), \
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F'), \
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F') ]

    _wiglib.DJ_m_k.restype = None

    _wiglib.DJ_m_k( J_c, m_c, npoints_c, grid_3d, Dr, Di )

    wig = Dr.reshape((npoints_3d,2*J+1)) + Di.reshape((npoints_3d,2*J+1))*1j # (ipoint,k)

    return wig


def DJ_m_k_2D(J, m, grid):
    """Computes Wigner D-functions d_{m,k}^{(J)}e^{-ik*chi} on a grid of Euler angles theta and chi, for k=-J..J and selected m.

    Parameters
    ----------
    J : int
       Value of J quantum number.
    m : int
       Value of m quantum number.
    grid : array (2,npoints)
        Numpy array containing grid of Euler angles, grid[:2,ipoint] = (theta,chi)
    Returns
    -------
    wig : array (npoints, 2*J+1)
        Contains values of D-functions on grid, d_{m,k}^{(J)}e^{-ik*chi} = wig[ipoint,k+J]
    """

    npoints_2d = grid.shape[1]
    grid_2d = np.asfortranarray(grid)

    Dr = np.asfortranarray(np.zeros((npoints_2d,2*J+1), dtype=np.float64)) # (ipoint,k)
    Di = np.asfortranarray(np.zeros((npoints_2d,2*J+1), dtype=np.float64))

    npoints_c = c_int(npoints_2d)
    J_c = c_int(J)
    m_c = c_int(m)

    _wiglib.DJ_m_k_2D.argtypes = [ c_int, c_int, c_int, \
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F'), \
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F'), \
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F') ]

    _wiglib.DJ_m_k_2D.restype = None

    _wiglib.DJ_m_k_2D( J_c, m_c, npoints_c, grid_2d, Dr, Di )

    wig = Dr.reshape((npoints_2d,2*J+1)) + Di.reshape((npoints_2d,2*J+1))*1j # (ipoint,k)

    return wig