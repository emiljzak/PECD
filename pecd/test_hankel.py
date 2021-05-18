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

#Hankel
import hankel
from hankel import HankelTransform  as HT   # Import the basic class


#PyHank
import scipy.special as scipy_bessel
from pyhank import qdht, iqdht, HankelTransform

print("Using hankel v{}".format(hankel.__version__))


def compareHanks():
    a = 3
    radius = np.linspace(0, 3, 1024)
    f = np.exp(-a ** 2 * radius ** 2)
    fan = lambda radius: np.exp(-a ** 2 * radius ** 2)
    kr, actual_ht = qdht(radius, f)
    expected_ht = 2*np.pi*(1 / (2 * a**2)) * np.exp(-kr**2 / (4 * a**2))
    assert np.allclose(expected_ht, actual_ht)


    h    = HT(0, N=5000, h=0.0001)        # Create the HankelTransform instance, order zero
    hhat = h.transform(fan,radius,ret_err=False) 

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Gaussian function')
    plt.plot(radius, f)
    plt.xlabel('Radius /$r$')
    plt.subplot(2, 1, 2)
    plt.plot(kr, expected_ht, label='Analytical')
    plt.plot(kr, actual_ht, marker='x', linestyle='None', label='QDHT')
    plt.plot(kr,hhat,label='from Hankel')
    plt.title('Hankel transform - also Gaussian')
    plt.xlabel('Frequency /$v$')
    plt.xlim([0, 50])
    plt.legend()
    plt.tight_layout()
    plt.show()


def hank():

    Lmax = 2

    ncontours = 20
    npoints = 100
    rmax    = 1.0 * np.pi
    rmin    = 0.01

    r = np.linspace(0,1,1000)       # Define a physical grid
    k = np.logspace(-2,2,1000)           # Define a spectral grid
    f    = lambda r : np.sin(r)/(r**2 + 1)

    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r"$F_0(k)$", fontsize=15)
    plt.xlabel(r"$k$", fontsize=15)
    for deg in range(0,1):
        h    = HankelTransform(nu=deg,N=2000,h=0.0001)        # Create the HankelTransform instance, order zero
        hhat = h.transform(f,k,ret_err=False) 
        hhat_sp = spline(k, hhat)  
        hhat_inv = h.transform(hhat_sp, r, False, inverse=True)  
        plt.plot(r,f(r))
        plt.plot(r,hhat_inv)

    plt.show()

    #hf = ht.integrate(f)
    #print(hf)
    """
    r_grid = np.linspace(rmin, rmax, npoints, endpoint=True, dtype=float)

 
    f = lambda r:  np.sin(r**2)

    val = np.zeros((len(x3d),len(y3d),len(z3d)), dtype = complex)
    
    start_time = time.time()



   
    ft_grid = np.linspace(-1.0/(rmax), 1.0/(rmax), npoints, endpoint=True, dtype=float)

    yftgrid, zftgrid = np.meshgrid(ft_grid,ft_grid)

    fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    axft = fig.add_subplot(spec[0, 0])

    line_ft = axft.contourf(yftgrid, zftgrid , ftval[50,:npoints,:npoints].real/np.max(np.abs(ftval)), 
                                        ncontours, cmap = 'jet', vmin=-0.2, vmax=0.2) #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    plt.colorbar(line_ft, ax=axft, aspect=30)
    plt.legend()   
    plt.show()  

    """
#hank()
compareHanks()