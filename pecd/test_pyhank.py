import numpy as np
from scipy.fftpack import fftn
import time
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from scipy.interpolate import InterpolatedUnivariateSpline as spline 
import scipy.special
import scipy.special as scipybessel

import numpy.ma as ma

#PyHank
from pyhank import qdht, iqdht, HankelTransform

import hankel
from hankel import HankelTransform as HT        


# 1D Gaussian function
def gauss1d(x, x0, fwhm):
    return np.exp(-2 * np.log(2) * ((x - x0) / fwhm) ** 2)


# Plotting function equivalent to Matlab's imagesc
def imagesc(x: np.ndarray, y: np.ndarray, intensity: np.ndarray, axes=None, **kwargs):
    assert x.ndim == 1 and y.ndim == 1, "Both x and y must be 1d arrays"
    assert intensity.ndim == 2, "Intensity must be a 2d array"
    extent = (x[0], x[-1], y[-1], y[0])
    if axes is None:
        img = plt.imshow(intensity, extent=extent, **kwargs, aspect='auto')
    else:
        img = axes.imshow(intensity, extent=extent, **kwargs, aspect='auto')
    img.axes.invert_yaxis()
    return img


def testpyhank():
    transformer = HankelTransform(order=0, max_radius=100, n_points=1024)
    f = scipy.special.jv(1, transformer.r) / transformer.r
    ht = transformer.qdht(f)

    plt.figure()
    plt.plot(transformer.kr, ht)
    plt.xlim([0, 5])
    plt.xlabel('Radial wavevector /m$^{-1}$')

    plt.show()

def hankel_transform_of_sinc(v,gamma,p):
    ht = np.zeros_like(v)
    ht[v < gamma] = (v[v < gamma] ** p * np.cos(p * np.pi / 2)
                     / (2 * np.pi * gamma * np.sqrt(gamma ** 2 - v[v < gamma] ** 2)
                        * (gamma + np.sqrt(gamma ** 2 - v[v < gamma] ** 2)) ** p))
    ht[v >= gamma] = (np.sin(p * np.arcsin(gamma / v[v >= gamma]))
                      / (2 * np.pi * gamma * np.sqrt(v[v >= gamma] ** 2 - gamma ** 2)))
    return ht

def sinc(x):
    return np.sin(x) / x

def testpyhank2():
    for p in range(0,4):
        
        some_grid = np.linspace(0.01,3,400)
        some_other_grid = np.linspace(0.01,3,200)
        transformer = HankelTransform(p, radial_grid=some_grid)
        
        gamma = 5
        

        func = sinc(2 * np.pi * gamma * some_grid)
        
        func_int = transformer.to_transform_r(func)

        print(transformer.v)
        print(transformer.kr)
        #print(func.shape)
        #print(func_int.shape)
        #exit()
        #print(func_int)

        plt.plot(transformer.r , func_int, label='f_int')
        plt.plot(transformer.r , func, label='f')
        plt.show()

        expected_ht = hankel_transform_of_sinc(transformer.v,gamma,p)

        ht = transformer.qdht(func)


        #vgrid = ma.masked_where(transformer.v > 10,transformer.v  )

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(transformer.v , expected_ht, label='Analytical')
        plt.plot(transformer.v, ht, marker='+', linestyle='None', label='QDHT')
        plt.title(f'Hankel Transform, p={p}')
        plt.legend()

        plt.tight_layout()
        plt.show()



def compare_hanks():

    some_grid = np.linspace(-10,10,400)
         # Define a spectral grid
    gamma = 5
    #func = sinc(2 * np.pi * gamma * some_grid)
    func = gauss1d(some_grid, 0.0, 1.0)
    func_int = spline(some_grid, func)     



    #plt.figure()
    #plt.subplot(2, 1, 1)
    #plt.plot(some_grid , func, label='func')
    #plt.plot(some_grid, func_int(some_grid), marker='+', linestyle='None', label='func_int')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    #pyHank part
    pyHank_H  = HankelTransform(0, radial_grid=some_grid)

    #with np.printoptions(precision=4, suppress=True, formatter={'float': '{:10.3f}'.format}, linewidth=400):
    print(some_grid)
    print(pyHank_H.r)

    func_res = pyHank_H.to_transform_r(func)
    pyHank_ht_int = pyHank_H.qdht(func_int(pyHank_H.r))
    pyHank_ht_original = pyHank_H.qdht(func_res)

    #Hankel part
    Hankel_H    = HT(nu=0, N=2000, h=0.001)        # Create the HankelTransform instance, order zero
    Hankel_ht   = Hankel_H.transform(func_int, pyHank_H.v*2.0*np.pi, ret_err=False)    

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(pyHank_H.v , pyHank_ht_int, label='pyHank_int')
    plt.plot(pyHank_H.v , pyHank_ht_original, label='pyHank_original',marker='+')
    plt.plot(pyHank_H.v, 2.0*np.pi * Hankel_ht, marker='+', linestyle='None', label='Hankel')
    plt.legend()
    plt.tight_layout()
    plt.show()


#testpyhank2()

compare_hanks()