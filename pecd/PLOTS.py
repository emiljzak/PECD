from mayavi import mlab

import numpy as np
from scipy.special import sph_harm

import os

import input
import GRID

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker

def spharm(l,m,theta,phi):
    return sph_harm(m, l, phi, theta)
    
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

def calc_wf_xyzgrid(nlobs,nbins,ivec,Gr,wffile,grid):
    coeffs = read_coeffs(wffile,nvecs=5)

    xl=np.zeros(nlobs)
    w=np.zeros(nlobs)
    xl,w=GRID.gauss_lobatto(nlobs,14)
    w=np.array(w)

    X, Y, Z = grid[0], grid[1], grid[2]
    print(X.shape)
    val = np.zeros((X.shape[0]*X.shape[0]*X.shape[0]))


    rx,thetax,phix = cart2sph(X,Y,Z)
    r= rx.flatten()
    theta =thetax.flatten()
    phi =phix.flatten()

    print(theta.shape)
    for ipoint in coeffs:
        print(ipoint)
        if np.abs(ipoint[5][ivec]) > 1e-2:
            val +=  ipoint[5][ivec] * chi(ipoint[0], ipoint[1],r,Gr,w,nlobs,nbins) * spharm(ipoint[3], ipoint[4], theta, phi).real / r

    return val/ np.max(val)

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

def plot_wf_angrad(rmin,rmax,npoints,coeffs,rgrid,nlobs,nbins,ivec):
    #================ radial-angular in real space ===============#

    """plot the selected wavefunctions functions"""
    """ Only radial part is plotted"""

    fig = plt.figure(figsize=(2, 2), dpi=200, constrained_layout=True)

    spec = gridspec.GridSpec(ncols=1, nrows=1,figure=fig)
    axradang_r = fig.add_subplot(spec[0, 0],projection='polar')

    rang = np.linspace(rmin,rmax,npoints,endpoint=True,dtype=float)
    gridtheta1d = 2 * np.pi * rang
    rmesh, thetamesh = np.meshgrid(rang, gridtheta1d)

    x=np.zeros(nlobs)
    w=np.zeros(nlobs)
    x,w=GRID.gauss_lobatto(nlobs,14)
    w=np.array(w)
    x=np.array(x) # convert back to np arrays
    nprint = 1 #how many functions to print

    y = np.zeros((len(rang),len(rang)),dtype=complex)

    phi0 = 0.0

    #counter = 0

    for ipoint in coeffs:
        if np.abs(ipoint[5][ivec]) > 1e-6:
            print(ipoint)
            for i in range(len(rang)):
                y[i,:] +=  ipoint[5][ivec] * chi(ipoint[0],ipoint[1],rang[:],rgrid,w,nlobs,nbins) * spharm(ipoint[3], ipoint[4], gridtheta1d[i], phi0).real

    line_angrad_r = axradang_r.contourf(thetamesh, rmesh,  rmesh*np.abs(y)/np.max(rmesh*np.abs(y)),20,cmap = 'jet') #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    plt.colorbar(line_angrad_r, ax=axradang_r, aspect=30)

    plt.legend()   
    plt.show()     

def plot_wf_isosurf(nlobs,nbins,Gr,wffile):
    mlab.clf()
    fig = mlab.figure(1, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1200, 1200))
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 0.0 , 0.0, 0.0], distance=10.0, figure=fig)

    plot_molecule = False

    if plot_molecule == True:
        # The position of the atoms
        scale = 40.0 / 5.5
        trans = np.array([0.0, 0.0, 0.0])
        atoms_O = np.array([0.0, 0.0, 0.0]) * scale + trans
        atoms_H1 = np.array([0.757,   0.586,    0.0000]) * scale + trans
        atoms_H2 = np.array([ -0.757,    0.586,     0.000]) * scale + trans



        O = mlab.points3d(atoms_O[0], atoms_O[1], atoms_O[2],
                        scale_factor=3,
                        resolution=20,
                        color=(1, 0, 0),
                        scale_mode='none',
                        figure=fig,
                        mode='sphere')

        H1 = mlab.points3d(atoms_H1[0], atoms_H1[1], atoms_H1[2],
                        scale_factor=2,
                        resolution=20,
                        color=(1, 1, 1),
                        scale_mode='none',
                        figure=fig,
                        mode='sphere')

        H2 = mlab.points3d(atoms_H2[0], atoms_H2[1], atoms_H2[2],
                        scale_factor=2,
                        resolution=20,
                        color=(1, 1, 1),
                        scale_mode='none',
                        figure=fig,
                        mode='sphere')

    npts = 40j
    grange = 5.0
    
    xmax = grange
    xmin = -1.0 * grange
    
    zmax = grange
    zmin = -1.0 * grange
    
    ymax = grange
    ymin = -1.0 * grange
    
    ivec = 2

    x, y, z = np.mgrid[xmin:xmax:npts, ymin:ymax:npts, zmin:zmax:npts]
    #wf = np.sin(x**2 + y**2 + 2. * z**2)
    wf = calc_wf_xyzgrid(nlobs,nbins,ivec,Gr,wffile,[x,y,z])
    fmin = wf.min()
    fmax = wf.max()
    print(wf)
    wf2 = wf.reshape(int(np.abs(npts)),int(np.abs(npts)),int(np.abs(npts)))
    print(wf2)
    #plot volume
    #mlab.pipeline.volume(mlab.pipeline.scalar_field(wf),  vmin=fmin + 0.65 * (fmax - fmin),
    #                               vmax=fmin + 0.9 * (fmax - fmin))
    
    #plot isosurface
    mywf = mlab.contour3d(wf2, contours=[0.05,0.1,0.2], colormap='gnuplot',opacity=0.5) #[0.9,0.7,0.5,0.4]
    mlab.view(132, 54, 45, [21, 20, 21.5])  
    mlab.show()

def plot_wf_volume(nlobs,nbins,Gr,wffile):
    mlab.clf()
    fig = mlab.figure(1, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1200, 1200))
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 0.0 , 0.0, 0.0], distance=10.0, figure=fig)

    plot_molecule = False

    if plot_molecule == True:
        # The position of the atoms
        scale = 40.0 / 5.5
        trans = np.array([0.0, 0.0, 0.0])
        atoms_O = np.array([0.0, 0.0, 0.0]) * scale + trans
        atoms_H1 = np.array([0.757,   0.586,    0.0000]) * scale + trans
        atoms_H2 = np.array([ -0.757,    0.586,     0.000]) * scale + trans



        O = mlab.points3d(atoms_O[0], atoms_O[1], atoms_O[2],
                        scale_factor=3,
                        resolution=20,
                        color=(1, 0, 0),
                        scale_mode='none',
                        figure=fig,
                        mode='sphere')

        H1 = mlab.points3d(atoms_H1[0], atoms_H1[1], atoms_H1[2],
                        scale_factor=2,
                        resolution=20,
                        color=(1, 1, 1),
                        scale_mode='none',
                        figure=fig,
                        mode='sphere')

        H2 = mlab.points3d(atoms_H2[0], atoms_H2[1], atoms_H2[2],
                        scale_factor=2,
                        resolution=20,
                        color=(1, 1, 1),
                        scale_mode='none',
                        figure=fig,
                        mode='sphere')

    npts = 40j
    grange = 5.0
    
    xmax = grange
    xmin = -1.0 * grange
    
    zmax = grange
    zmin = -1.0 * grange
    
    ymax = grange
    ymin = -1.0 * grange
    
    ivec = 2

    x, y, z = np.mgrid[xmin:xmax:npts, ymin:ymax:npts, zmin:zmax:npts]
    #wf = np.sin(x**2 + y**2 + 2. * z**2)
    wf = calc_wf_xyzgrid(nlobs,nbins,ivec,Gr,wffile,[x,y,z])
    fmin = wf.min()
    fmax = wf.max()

    wf2 = wf.reshape(int(np.abs(npts)),int(np.abs(npts)),int(np.abs(npts)))

    #plot volume
    mlab.pipeline.volume(mlab.pipeline.scalar_field(wf2),  vmin=fmin + 0.65 * (fmax - fmin),
                                   vmax=fmin + 0.9 * (fmax - fmin))
    

    mlab.view(132, 54, 45, [21, 20, 21.5])  
    mlab.show()

def plot_snapshot(params,maparray,rgrid,ivec):
    #make it general
    nlobs = params['nlobs']
    nbins = params['bound_nbins'] + params['nbins']
    npoints = 200
    rmax    = nbins * params['bound_binw']
    
    figsize = (3,3)
    dpi     = 200
    fig = plt.figure(figsize, dpi, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    if params['plot_types']['r-radial_angular'] == True:
        #================ radial-angular in real space ===============#
        ax_radang_r = fig.add_subplot(spec[0, 0],projection='polar')
        width = 0.0
        for elem in params['FEMLIST']:
            width += elem[0] * elem[2]



    #read the wavepacket

    #align with maparray

    #merge into coeffs

        PLOTS.plot_wf_angrad(rmin,rmax,npoints,coeffs,rgrid,nlobs,nbins,ivec)

    if params["save_snapthots"] == True:
        fig.savefig("angrad_t=" + str("%4.1f"%(t/np.float64(1.0/24.188))) + "_.pdf", bbox_inches='tight')


if __name__ == "__main__":      

    # preparation of parameters
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


    params = input.gen_input()
    wffile = params['working_dir'] + "psi_init_h2o_3_24_20.0_2_uhf_631Gss.dat"#+ "psi0_h2o_1_20_10.0_4_uhf_631Gss.dat"
    nvecs = 7 #how many vectors to load?
    ivec = 0 #which vector to plot?
    coeffs = read_coeffs(wffile,nvecs)
    nbins = params['bound_nbins'] + params['nbins']

    Gr, Nr = GRID.r_grid( params['bound_nlobs'], nbins , params['bound_binw'],  params['bound_rshift'] )


    """ plot radial basis """
    #plot_chi(0.0,params['bound_binw']*params['bound_nbins'],1000,Gr,params['bound_nlobs'], params['bound_nbins'])
    #exit()

    """ plot radial wavefunction at a given ray=(theta,phi) """

    #plot_wf_rad(0.0, params['bound_binw']*params['bound_nbins'], 1000, coeffs ,\
    #            Gr, params['bound_nlobs'], params['bound_nbins'],nvecs)

    """ plot angular wavefunction at a given distance """
    #r0 = 2.0
    #plot_wf_ang(r0,coeffs,Gr,params['bound_nlobs'], params['bound_nbins'])

    """ plot angular-radial wavefunction on a polar plot"""
    plot_wf_angrad( 0.0, params['bound_binw'] * nbins, 200, coeffs ,\
                    Gr, params['bound_nlobs'], nbins, ivec)

    """ plot 3D isosurface of the wavefunction amplitude (density)"""
    #plot_wf_isosurf(params['bound_nlobs'], params['bound_nbins'],Gr,wffile)


    """ plot 4D volume of the wavefunction amplitude (density)"""
    #plot_wf_volume(params['bound_nlobs'], params['bound_nbins']+params['nbins'],Gr,wffile)
    exit()

