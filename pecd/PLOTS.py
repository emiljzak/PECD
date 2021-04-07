# Create the data.
from numpy import pi, sin, cos, mgrid
from mayavi import mlab
import numpy as np
import os
import input
import GRID
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker


def myavitest():
    dphi, dtheta = pi/250.0, pi/250.0
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    m0 = 4
    m1 = 3
    m2 = 2
    m3 = 3
    m4 = 6
    m5 = 2
    m6 = 6
    m7 = 4
    r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
    x = r*sin(phi)*cos(theta)
    y = r*cos(phi)
    z = r*sin(phi)*sin(theta)

    # View it.

    s = mlab.mesh(x, y, z)
    mlab.show()


def read_coeffs(filename,nvecs):

    coeffs = []
    fl = open( filename , 'r' )
    for line in fl:
        words   = line.split()
        i       = int(words[0])
        n       = int(words[1])
        xi      = int(words[2])
        l       = int(words[3])
        m       = int(words[4])
        c       = []
        for ivec in range(nvecs):
            c.append(float(words[5+ivec]))
        coeffs.append([i,n,xi,l,m,np.asarray(c)])
    return coeffs

def spharm(l,m,theta,phi):
    return sph_harm(m, l, phi, theta)

    


def plot_wf_rad(rmin,rmax,npoints,coeffs,rgrid,nlobs,nbins,nvecs):
    """plot the selected wavefunctions functions"""
    """ Only radial part is plotted"""

    r = np.linspace(rmin,rmax,npoints,endpoint=True,dtype=float)

    x=np.zeros(nlobs)
    w=np.zeros(nlobs)
    x,w=GRID.gauss_lobatto(nlobs,14)
    w=np.array(w)
    x=np.array(x) # convert back to np arrays
    nprint = 1 #how many functions to print

    y = np.zeros(len(r))
    theta0 = 0.0
    phi0 = 0.0

    #counter = 0

    for ivec in range(nvecs):
        y = np.zeros(len(r))
        for ipoint in coeffs:

            y +=  ipoint[5][ivec] * chi(ipoint[0],ipoint[1],r,rgrid,w,nlobs,nbins) * spharm(ipoint[3], ipoint[4], theta0, phi0).real


        plt.plot(r, np.abs(y)**2/max(np.abs(y)**2), '-',label=str(ivec)) 

    plt.xlabel('r/a.u.')
    plt.ylabel('Radial eigenfunction')
    plt.legend()   
    plt.show()     

def chi(i,n,r,rgrid,w,nlobs,nbins):
    # r is the argument f(r)
    # rgrid is the radial grid rgrid[i][n]
    # w are the unscaled lobatto weights

    w /=sum(w[:]) #normalization!!!
    val=np.zeros(np.size(r))
    
    if n==0 and i<nbins-1: #bridge functions
        #print("bridge: ", n,i)
        val = ( f(i,nlobs-1,r,rgrid,nlobs,nbins) + f(i+1,0,r,rgrid,nlobs,nbins) ) * np.sqrt( float( w[nlobs-1] ) + float( w[0] ) )**(-1)
        #print(type(val),np.shape(val))
        return val 
    elif n>0 and n<nlobs-1:
        val = f(i,n,r,rgrid,nlobs,nbins) * np.sqrt( float( w[n] ) ) **(-1) 
        #print(type(val),np.shape(val))
        return val
    else:
        return val

def f(i,n,r,rgrid,nlobs,nbins): 
    """calculate f_in(r). Input r can be a scalar or a vector (for quadpy quadratures) """
    
    #print("shape of r is", np.shape(r), "and type of r is", type(r))

    if np.isscalar(r):
        prod=1.0
        if  r>= rgrid[i][0] and r <= rgrid[i][nlobs-1]:
            for mu in range(0,nlobs):
                if mu !=n:
                    prod*=(r-rgrid[i][mu])/(rgrid[i][n]-rgrid[i][mu])
            return prod
        else:
            return 0.0

    else:
        prod=np.ones(np.size(r), dtype=float)
        for j in range(0,np.size(r)):

            for mu in range(0,nlobs):
                if mu !=n:
                    prod[j] *= (r[j]-rgrid[i,mu])/(rgrid[i,n]-rgrid[i,mu])
                else:
                    prod[j] *= 1.0

    return prod


def plot_chi(rmin,rmax,npoints,rgrid,nlobs,nbins):
    """plot the selected radial basis functions"""
    r = np.linspace(rmin,rmax,npoints,endpoint=True,dtype=float)


    x=np.zeros(nlobs)
    w=np.zeros(nlobs)
    x,w=GRID.gauss_lobatto(nlobs,14)
    w=np.array(w)
    x=np.array(x) # convert back to np arrays
    nprint = 1 #how many functions to print

    y = np.zeros((len(r),nlobs * nbins))

    counter = 0
    wcounter = 0
    for i in range(nbins):
        for n in range(nlobs):
            y[:,counter] = chi(i,n,r,rgrid,w,nlobs,nbins) #* w[wcounter]**(0.5)
            counter += 1
        wcounter +=1

    figLob = plt.figure()
    plt.xlabel('r/a.u.')
    plt.ylabel('Lobatto basis function')
    plt.legend()   

    plt.plot(r, y) 
    plt.show()     


def plot_wf_ang(r0,coeffs,rgrid, nlobs,nbins):
    """plot the angular basis"""
    theta_1d = np.linspace(0,   np.pi,  2*91) # 
    phi_1d   = np.linspace(0, 2*np.pi, 2*181) # 

    theta_2d, phi_2d = np.meshgrid(theta_1d, phi_1d)
    xyz_2d = np.array([np.sin(theta_2d) * np.sin(phi_2d), np.sin(theta_2d) * np.cos(phi_2d), np.cos(theta_2d)]) #2D grid of cartesian coordinates

    colormap = cm.ScalarMappable( cmap=plt.get_cmap("cool") )
    colormap.set_clim(-.45, .45)
    limit = .5

    plt.figure()
    ax = plt.gca(projection = "3d")
    
    x=np.zeros(nlobs)
    w=np.zeros(nlobs)
    x,w=GRID.gauss_lobatto(nlobs,14)
    w=np.array(w)
    x=np.array(x) # convert back to np arrays
    nprint = 1 #how many functions to print

    #counter = 0

    y = np.zeros((362,182))
    for ipoint in coeffs:

        y +=  ipoint[5][2] * chi(ipoint[0],ipoint[1],r0,rgrid,w,nlobs,nbins) * spharm(ipoint[3], ipoint[4], theta_2d, phi_2d).real

    r = np.abs(y.real)*xyz_2d #calculate a point in 3D cartesian space for each value of spherical harmonic at (theta,phi)
    
    ax.plot_surface(r[0], r[1], r[2], facecolors=colormap.to_rgba(y.real), rstride=1, cstride=1)
    ax.set_xlim(-limit,limit)
    ax.set_ylim(-limit,limit)
    ax.set_zlim(-limit,limit)
    #ax.set_aspect("equal")
    #ax.set_axis_off()
    
            
    plt.show()


import numpy
from mayavi.mlab import *

def sph2cart(r,theta,phi):
    x=r*np.sin(theta)*np.cos(phi)
    y=r*np.sin(theta)*np.sin(phi)
    z=r*np.cos(theta)
    return x,y,z

def cart2sph(x,y,z):
    r=np.sqrt(x**2+y**2+z**2)
    theta=np.arctan(np.sqrt(x**2+y**2)/z)
    phi=np.arctan(y/x)
    return r,theta,phi

def calc_wf_xyzgrid(nlobs,nbins,ivec,Gr,wffile,grid):
    coeffs = read_coeffs(wffile,nvecs=4)

    xl=np.zeros(nlobs)
    w=np.zeros(nlobs)
    xl,w=GRID.gauss_lobatto(nlobs,14)
    w=np.array(w)

    X, Y, Z = grid[0], grid[1], grid[2]
    val = np.zeros((X.shape[0],1,1))
    print(X.shape)

    r,theta,phi = cart2sph(X,Y,Z)
    #r= rx.flatten()
    #theta =thetax.flatten()
    #phi =phix.flatten()
    print(theta.shape)
    for ipoint in coeffs:

        val +=  ipoint[5][ivec] * spharm(ipoint[3], ipoint[4], theta, phi).real / r
    return val

def test_contour3d(nlobs,nbins,Gr,wffile):
    # The position of the atoms
    atoms_x = np.array([1.0, 0.0, 0.0]) * 40 /5.5
    atoms_y = np.array([0.0, 1.0, 0.0]) * 40 /5.5
    atoms_z = np.array([0.0, 0.0, 0.0]) * 40 /5.5

    #atoms_x = np.array([2.9, 2.9, 3.8]) * 40 /5.5
    #atoms_y = np.array([3.0, 3.0, 3.0]) * 40 /5.5
    #atoms_z = np.array([3.8, 2.9, 2.7]) * 40 /5.5

    O = mlab.points3d(atoms_x[1:-1], atoms_y[1:-1], atoms_z[1:-1],
                    scale_factor=3,
                    resolution=20,
                    color=(1, 0, 0),
                    scale_mode='none')

    H1 = mlab.points3d(atoms_x[:1], atoms_y[:1], atoms_z[:1],
                    scale_factor=2,
                    resolution=20,
                    color=(1, 1, 1),
                    scale_mode='none')

    H2 = mlab.points3d(atoms_x[-1:], atoms_y[-1:], atoms_z[-1:],
                    scale_factor=2,
                    resolution=20,
                    color=(1, 1, 1),
                    scale_mode='none')
    # The bounds between the atoms, we use the scalar information to give
    # color
    mlab.plot3d(atoms_x, atoms_y, atoms_z, [1, 2, 1],
                tube_radius=0.4, colormap='Reds')


    npts = 100j
    xmax = 2.0
    x, y, z = np.ogrid[-1.0*xmax:xmax:npts, -1.0*xmax:xmax:npts, -1.0*xmax:xmax:npts]
    ivec = 2
    #wf = calc_wf_xyzgrid(nlobs,nbins,ivec,Gr,wffile,[x,y,z])
    wf = np.sin(x**2 + y**2 + 2. * z**2) / (z**2+x**2+y**2)
    mlab.pipeline.volume(mlab.pipeline.scalar_field(wf), vmin=0.0, vmax=0.9)

    show()
    #scalars =sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y + z) 
    #wf.reshape(10,10,10)
    #obj = contour3d(wf, contours=5, transparent=False)



def test_contour_surf():
    """Test contour_surf on regularly spaced co-ordinates like MayaVi."""
    def f(x, y):
        sin, cos = np.sin, np.cos
        return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

    x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
    s = contour_surf(x, y, f)
    show()
    return s



def chemistry():
    # Retrieve the electron localization data for H2O #############################
    import os
    if not os.path.exists('h2o-elf.cube'):
        # Download the data
        try:
            from urllib import urlopen
        except ImportError:
            from urllib.request import urlopen
        print('Downloading data, please wait')
        opener = urlopen(
            'http://code.enthought.com/projects/mayavi/data/h2o-elf.cube'
            )
        open('h2o-elf.cube', 'wb').write(opener.read())


    # Plot the atoms and the bonds ################################################
    import numpy as np
    from mayavi import mlab
    mlab.figure(1, bgcolor=(0, 0, 0), size=(350, 350))
    mlab.clf()

    # The position of the atoms
    atoms_x = np.array([2.9, 2.9, 3.8]) * 40 / 5.5
    atoms_y = np.array([3.0, 3.0, 3.0]) * 40 / 5.5
    atoms_z = np.array([3.8, 2.9, 2.7]) * 40 / 5.5

    O = mlab.points3d(atoms_x[1:-1], atoms_y[1:-1], atoms_z[1:-1],
                    scale_factor=3,
                    resolution=20,
                    color=(1, 0, 0),
                    scale_mode='none')

    H1 = mlab.points3d(atoms_x[:1], atoms_y[:1], atoms_z[:1],
                    scale_factor=2,
                    resolution=20,
                    color=(1, 1, 1),
                    scale_mode='none')

    H2 = mlab.points3d(atoms_x[-1:], atoms_y[-1:], atoms_z[-1:],
                    scale_factor=2,
                    resolution=20,
                    color=(1, 1, 1),
                    scale_mode='none')

    # The bounds between the atoms, we use the scalar information to give
    # color
    mlab.plot3d(atoms_x, atoms_y, atoms_z, [1, 2, 1],
                tube_radius=0.4, colormap='Reds')

    # Display the electron localization function ##################################
    fl = open('h2o-elf.cube','r')
    print(fl.readlines()[9:])
    # Load the data, we need to remove the first 8 lines and the '\n'
    str = ' '.join(fl.readlines()[9:])
    data = np.fromstring(str, sep=' ')
    data.shape = (40, 40, 40)

    source = mlab.pipeline.scalar_field(data)
    min = data.min()
    max = data.max()
    vol = mlab.pipeline.volume(source, vmin=min + 0.65 * (max - min),
                                    vmax=min + 0.9 * (max - min))

    mlab.view(132, 54, 45, [21, 20, 21.5])

    mlab.show()


os.environ['KMP_DUPLICATE_LIB_OK']='True'




params = input.gen_input()
wffile = params['working_dir'] + "psi0_h2o_1_20_10.0_4_uhf_631Gss.dat"

nvecs = 6

coeffs = read_coeffs(wffile,nvecs)

Gr, Nr = GRID.r_grid( params['bound_nlobs'], params['bound_nbins'], params['bound_binw'],  params['bound_rshift'] )
#plot_chi(0.0,params['bound_binw']*params['bound_nbins'],1000,Gr,params['bound_nlobs'], params['bound_nbins'])
#exit()

test_contour3d(params['bound_nlobs'], params['bound_nbins'],Gr,wffile)
exit()

r0 = 2.0
plot_wf_ang(r0,coeffs,Gr,params['bound_nlobs'], params['bound_nbins'])
exit()
plot_wf_rad(0.0, params['bound_binw']*params['bound_nbins'], 1000, coeffs ,\
            Gr, params['bound_nlobs'], params['bound_nbins'],nvecs)