from mayavi import mlab

import numpy as np
from scipy.special import sph_harm
from scipy import interpolate

import os

import input
import GRID
import MAPPING

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

def chi(i,n,r,Gr,w,nlobs,nbins):
    # r is the argument f(r)
    # rgrid is the radial grid rgrid[i][n]
    # w are the unscaled lobatto weights

    #w /= sum(w[:]) #normalization!!!
    val = np.zeros(np.size(r))
    
    if n == nlobs-1: #bridge functions
        #print("bridge: ", n,i)


        val = ( f(i,nlobs-1,r,Gr,nlobs,nbins) + f(i+1,0,r,Gr,nlobs,nbins) ) * np.sqrt( float( w[nlobs-1] ) + float( w[0] ) )**(-1)
    #print(type(val),np.shape(val))
        return val

    elif n < nlobs-1:

        val = f(i,n,r,Gr,nlobs,nbins) * np.sqrt( float( w[n] ) ) **(-1) 
        #print(type(val),np.shape(val))
        return val

    else:
        return val

def f(i,n,r,Gr,nlobs,nbins): 
    """calculate f_in(r). Input r can be a scalar or a vector (for quadpy quadratures) """
    
    #print("shape of r is", np.shape(r), "and type of r is", type(r))

    if np.isscalar(r):
        prod=1.0
        if  r>= Gr[i][0] and r <= Gr[i][nlobs-1]:
            for mu in range(0,nlobs):
                if mu !=n:
                    prod*=(r-Gr[i][mu])/(Gr[i][n]-Gr[i][mu])
            return prod
        else:
            return 0.0

    else:
        prod = np.ones(np.size(r), dtype=float)
        for j in range(0,np.size(r)):
            if r[j] >= Gr[i,0] and r[j] <= Gr[i,nlobs-1]:
                for mu in range(0,nlobs):
                    if mu !=n:
                        prod[j] *= (r[j] - Gr[i,mu]) / (Gr[i,n] - Gr[i,mu])
                    else:
                        prod[j] *= 1.0
            else:
                prod[j] = 0.0
    return prod

def plot_chi(rmin,rmax,npoints,rgrid,nlobs,nbins):
    """plot the selected radial basis functions"""
    r = np.linspace(rmin, rmax, npoints, endpoint=True, dtype=float)

    x=np.zeros(nlobs)
    w=np.zeros(nlobs)
    x,w=GRID.gauss_lobatto(nlobs,14)
    w=np.array(w)
    x=np.array(x) # convert back to np arrays

    y = np.zeros((len(r), nbins * nlobs))

    counter = 0
    wcounter = 0
    for i in range(nbins):
        for n in range(nlobs-1):
            y[:,counter] = chi(i, n, r, rgrid ,w, nlobs, nbins) #* w[wcounter]**(0.5)
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
    #for ipoint in coeffs:

        #y  = spharm(ipoint[3], ipoint[4], theta_2d, phi_2d).real #+=  ipoint[5][2] * chi(ipoint[0],ipoint[1],r0,rgrid,w,nlobs,nbins) *
    y = spharm(1, 0, theta_2d, phi_2d).real
    r = np.abs(y.real)*xyz_2d #calculate a point in 3D cartesian space for each value of spherical harmonic at (theta,phi)
    
    ax.plot_surface(r[0], r[1], r[2], facecolors=colormap.to_rgba(y.real), rstride=1, cstride=1)
    ax.set_xlim(-limit,limit)
    ax.set_ylim(-limit,limit)
    ax.set_zlim(-limit,limit)
    #ax.set_aspect("equal")
    #ax.set_axis_off()
    
            
    plt.show()


def plot_wf_angrad_int_XZ(rmin,rmax,npoints,nlobs,nbins,psi,maparray,Gr,params,t,flist):
    #================ radial-angular in real space ===============#

    coeff_thr = 1e-5
    ncontours = 100

    fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    axradang_r = fig.add_subplot(spec[0, 0], projection='polar')

    rang = np.linspace(rmin, rmax, npoints, endpoint=True, dtype=float)
    unity_vec = np.linspace(0.0, 1.0, npoints, endpoint=True, dtype=float)
    gridtheta1d = 2 * np.pi * unity_vec

    rmesh, thetamesh = np.meshgrid(rang, gridtheta1d)

    x   =  np.zeros(nlobs)
    w   =  np.zeros(nlobs)
    x,w =  GRID.gauss_lobatto(nlobs,14)
    w   =  np.array(w)
    x   =  np.array(x) # convert back to np arrays

    y = np.zeros((len(rang),len(rang)), dtype=complex)

    phi0 = 0.0 * np.pi/2 #fixed polar angle
    #here we can do phi-averaging
    
    for ielem, elem in enumerate(maparray):
        if np.abs(psi[ielem]) > coeff_thr:
            #print(str(elem) + str(psi[ielem]))

            chir = flist[elem[2]-1](rang) #labelled by xi

            for i in range(len(rang)):
                y[i,:] +=  psi[ielem]  * spharm(elem[3], elem[4], gridtheta1d[i], phi0) * \
                           chir #chi(elem[0], elem[1], rang[:], Gr, w, nlobs, nbins) 

    line_angrad_r = axradang_r.contourf(thetamesh, rmesh, np.abs(y)/np.max(np.abs(y)), 
                                        ncontours, cmap = 'jet', vmin=0.0, vmax=0.5) #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    #plt.colorbar(line_angrad_r, ax=axradang_r, aspect=30)
    axradang_r.set_rlabel_position(100)
    #axradang_r.set_yticklabels(list(str(np.linspace(rmin,rmax,5.0)))) # set radial tick label
    #plt.legend()   
    #plt.show()  
    if params["save_snapthots"] == True:
        fig.savefig( params['job_directory']  + "/animation/" + "angrad_XZ_t=" +\
                     str("%4.1f"%(t/np.float64(1.0/24.188)))+"_.png" ,\
                     bbox_inches='tight')
    plt.close()

def plot_wf_angrad_int_XY(rmin,rmax,npoints,nlobs,nbins,psi,maparray,Gr,params,t,flist):
    #================ radial-angular in real space ===============#

    coeff_thr = 1e-5
    ncontours = 100

    fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    axradang_r = fig.add_subplot(spec[0, 0], projection='polar')

    rang = np.linspace(rmin, rmax, npoints, endpoint=True, dtype=float)
    unity_vec = np.linspace(0.0, 1.0, npoints, endpoint=True, dtype=float)
    gridphi1d = 2 * np.pi * unity_vec

    rmesh, thetamesh = np.meshgrid(rang, gridphi1d)

    x   =  np.zeros(nlobs)
    w   =  np.zeros(nlobs)
    x,w =  GRID.gauss_lobatto(nlobs,14)
    w   =  np.array(w)
    x   =  np.array(x) # convert back to np arrays

    y = np.zeros((len(rang),len(rang)), dtype=complex)

    theta0 = np.pi/2 #fixed azimuthal angle
    #here we can do phi-averaging
    
    for ielem, elem in enumerate(maparray):
        if np.abs(psi[ielem]) > coeff_thr:
            #print(str(elem) + str(psi[ielem]))

            chir = flist[elem[2]-1](rang) #labelled by xi

            for i in range(len(rang)):
                y[i,:] +=  psi[ielem]  * spharm(elem[3], elem[4], theta0, gridphi1d[i]) * \
                           chir #chi(elem[0], elem[1], rang[:], Gr, w, nlobs, nbins) 

    line_angrad_r = axradang_r.contourf(thetamesh, rmesh, np.abs(y)/np.max(np.abs(y)), 
                                        ncontours, cmap = 'jet', vmin=0.0, vmax=0.5) #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    #plt.colorbar(line_angrad_r, ax=axradang_r, aspect=30)
    axradang_r.set_rlabel_position(100)
    #axradang_r.set_yticklabels(list(str(np.linspace(rmin,rmax,5.0)))) # set radial tick label
    #plt.legend()   
    #plt.show()  
    if params["save_snapthots"] == True:
        fig.savefig( params['job_directory']  + "/animation/" + "angrad_XY_t=" +\
                     str("%4.1f"%(t/np.float64(1.0/24.188)))+"_.png" ,\
                     bbox_inches='tight')
    plt.close()


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

def plot_snapshot(params,psi,maparray,Gr,t):
    #make it general
    nlobs = params['bound_nlobs']
    nbins = params['bound_nbins'] + params['nbins']
    npoints = 60
    rmax    = nbins * params['bound_binw']

    #fig = plt.figure(figsize = (3.,3.), dpi=200, constrained_layout=True)
    #spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    #ax_radang_r = fig.add_subplot(spec[0, 0],projection='polar')
    if params['plot_types']['r-radial_angular'] == True:
        """================ radial-angular in real space ==============="""
        width = 0.0
        for elem in params['FEMLIST']:
            width += elem[0] * elem[2]

        plot_wf_angrad(0.0, rmax, npoints, nlobs, nbins, psi, maparray, Gr, params, t)

def plot_snapshot_int(params,psi,maparray,Gr,t,flist):
    #plot snapshots using interpolation for the chi functions
    #make it general
    nlobs = params['bound_nlobs']
    nbins = params['bound_nbins'] + params['nbins']
    npoints = 360
    rmax    = nbins * params['bound_binw']

    #fig = plt.figure(figsize = (3.,3.), dpi=200, constrained_layout=True)
    #spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    #ax_radang_r = fig.add_subplot(spec[0, 0],projection='polar')
    if params['plot_types']['r-radial_angular'] == True:
        """================ radial-angular in real space ==============="""
        width = 0.0
        for elem in params['FEMLIST']:
            width += elem[0] * elem[2]

        plot_wf_angrad_int_XZ(0.0, rmax, npoints, nlobs, nbins, psi, maparray, Gr, params, t, flist)
        plot_wf_angrad_int_XY(0.0, rmax, npoints, nlobs, nbins, psi, maparray, Gr, params, t, flist)
    
    
def interpolate_chi(Gr,nlobs,nbins,binw,maparray):

    w       =  np.zeros(nlobs)
    xx,w    =  GRID.gauss_lobatto(nlobs,14)
    w       =  np.array(w)

    interpolation_step = 0.05
    x = np.arange(0.0, nbins * binw + 0.10, interpolation_step)

    chilist  = []

    print(maparray)

    for i, elem in enumerate(maparray):
        chilist.append( interpolate.interp1d(x, chi(elem[0], elem[1], x, Gr, w, nlobs, nbins) ) )

    #xnew  = np.arange(0.02, nbins * binw, 0.01)
    #ynew = chilist[1](xnew)   # use interpolation function returned by `interp1d`
    #for s in range((nlobs-1) * nbins - 1):
    #    ynew = chilist[s](xnew)   # use interpolation function returned by `interp1d`
    #    plt.plot(x, chilist[s](x), 'o', xnew, ynew, '-')
    #plt.show()
    
    return chilist

def plot_elfield(Fvec,tgrid,time_to_au):
    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("time (as)")
    plt.ylabel("normalized Field components")
    ax.scatter(tgrid/time_to_au, -Fvec[2].real, label = "Field-x", marker = '.', color = 'r', s = 1)
    ax.scatter(tgrid/time_to_au, Fvec[0].imag, label = "Field-y", marker = '.', color = 'g', s = 1)
    ax.scatter(tgrid/time_to_au, Fvec[1], label = "Field-z", marker = '.', color = 'b', s = 1)
    ax.legend()
    plt.show()

def plot_2D_polar_map(func,grid_theta,kgrid,ncontours):
    fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    axft = fig.add_subplot(spec[0, 0], projection='polar')
    kmesh, thetamesh = np.meshgrid(kgrid,grid_theta)
    axft.set_ylim(0,1) #radial extent
    line_ft = axft.contourf(thetamesh, kmesh, func/np.max(func), 
                            ncontours, cmap = 'jet') #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    plt.colorbar(line_ft, ax=axft, aspect=30) 
    plt.legend()   
    plt.show()  



def plot_3D_angfunc(func,grid):
    #'func' on 'grid'

    print(func.shape)
    theta_2d, phi_2d = np.meshgrid(grid[:,1], grid[:,0])
    #theta_2d, phi_2d = grid[0], grid[1]
    xyz_2d = np.array([np.sin(theta_2d) * np.sin(phi_2d), np.sin(theta_2d) * np.cos(phi_2d), np.cos(theta_2d)]) #2D grid of cartesian coordinates

    colormap = cm.ScalarMappable( cmap=plt.get_cmap("cool") )
    colormap.set_clim(-.45, .45)
    limit = .5

    plt.figure()
    ax = plt.gca(projection = "3d")

    #counter = 0
    #y = np.zeros((362,182))
    #y = spharm(1, 0, theta_2d, phi_2d).real
    y = func / np.sqrt( 4.0 * np.pi /(2*1+1))
    r = np.abs(y.real)*xyz_2d #calculate a point in 3D cartesian space for each value of spherical harmonic at (theta,phi)
    print(r.shape)
    ax.plot_surface(r[0], r[1], r[2])
    ax.set_xlim(-limit,limit)
    ax.set_ylim(-limit,limit)
    ax.set_zlim(-limit,limit)
    #ax.set_aspect("equal")
    #ax.set_axis_off()
    
        
    plt.show()

def plot_spharm_rotated(D):
    # Create a sphere
    r = 0.3
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
    mlab.clf()
    # Represent spherical harmonics on the surface of the sphere

    m = 0 
    l = 2

    s = sph_harm(m, l, theta, phi).real

    s_rot = np.zeros((theta.shape[0],phi.shape[0]), dtype = complex)
    print(D.shape)

    for p in range(-l,l+1):
        s_rot += D[p+l,m+l] * sph_harm(p, l, theta, phi)

    s_rot = s_rot.real
    #mlab.mesh(x - m, y - n, z, scalars=s, colormap='jet')
    
    s[s < 0] *= 0.97

    s /= s.max()
    mlab.mesh(s * x - m, s * y - l, s * z + 1.3,
            scalars=s, colormap='Spectral')

    
    s_rot[s_rot < 0] *= 0.97

    s_rot /= s_rot.max()
    mlab.mesh(s_rot * x - m, s_rot * y - l, s_rot * z + 1.3,
            scalars=s_rot, colormap='Spectral')
    

    #mlab.view(90, 70, 6.2, (-1.3, -2.9, 0.25))
    mlab.show()


def plot_rotdens(rotdens):
    # Create a sphere
    r = 0.3
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0:pi:1000j, 0:2 * pi:1000j]

    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
    mlab.clf()
    # Represent spherical harmonics on the surface of the sphere


    s = rotdens
    #mlab.mesh(x - m, y - n, z, scalars=s, colormap='jet')
    
    s[s < 0] *= 0.97

    s /= s.max()
    mlab.mesh(s * x , s * y , s * z + 1.3,
            scalars=s, colormap='Spectral')

    #mlab.view(90, 70, 6.2, (-1.3, -2.9, 0.25))
    mlab.show()

if __name__ == "__main__":      

    # preparation of parameters
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


    params = input.gen_input()
    
    #plot 3D angular function
    #plot_3D_angfunc()
    #plot_spharm_rotated()
    exit()

    #wffile = params['working_dir'] + "psi0_h2o_1_12_30.0_4_uhf_631Gss.dat"# "psi_init_h2o_3_24_20.0_2_uhf_631Gss.dat"#+ "psi0_h2o_1_20_10.0_4_uhf_631Gss.dat"
    nvecs = 7 #how many vectors to load?
    ivec = params['ivec'] #which vector to plot?
    
    #coeffs = read_coeffs(wffile,nvecs)

    #psi = np.zeros(len(coeffs), dtype = complex)
    #print(psi.shape)
    #for ielem,elem in enumerate(coeffs):
    #    psi[ielem] = elem[5][ivec]

    
    nbins = params['bound_nbins'] + params['nbins']

    #Gr, Nr = GRID.r_grid( params['bound_nlobs'], nbins , params['bound_binw'],  params['bound_rshift'] )

    #maparray, Nbas = MAPPING.GENMAP( params['bound_nlobs'], params['bound_nbins'], params['bound_lmax'], \
    #                                 params['map_type'], params['working_dir'] )

    #maparray_chi, Nbas_chi = MAPPING.GENMAP_FEMLIST( params['FEMLIST'],  0, \
    #                                 params['map_type'], params['working_dir'] )


    #flist = interpolate_chi(Gr,params['bound_nlobs'],nbins,params['bound_binw'] ,maparray_chi)
    #exit()

    """ plot radial basis """
    #plot_chi(0.0,params['bound_binw']*params['bound_nbins'],1000,Gr,params['bound_nlobs'], params['bound_nbins'])
    #exit()

    """ plot radial wavefunction at a given ray=(theta,phi) """

    #plot_wf_rad(0.0, params['bound_binw']*params['bound_nbins'], 1000, coeffs ,\
    #            Gr, params['bound_nlobs'], params['bound_nbins'],nvecs)

    """ plot angular wavefunction at a given distance """
    r0 = 2.0
    coeffs = 0.0
    Gr   = 0.0
    plot_wf_ang(r0,coeffs,Gr,params['bound_nlobs'], params['bound_nbins'])

    """ plot angular-radial wavefunction on a polar plot"""
    #plot_snapshot(params,psi,maparray,Gr,t)
    #plot_wf_angrad( 0.0, params['bound_binw'] * params['bound_nbins'], 200,params['bound_nlobs'], nbins,\
    #                psi,maparray ,Gr, params, 0.0)
    #plot_wf_angrad(rmin,rmax,npoints,nlobs,nbins,psi,maparray,rgrid,params,t)
    """ plot 3D isosurface of the wavefunction amplitude (density)"""
    #plot_wf_isosurf(params['bound_nlobs'], params['bound_nbins'],Gr,wffile)


    """ plot 4D volume of the wavefunction amplitude (density)"""
    #plot_wf_volume(params['bound_nlobs'], params['bound_nbins']+params['nbins'],Gr,wffile)
    exit()

