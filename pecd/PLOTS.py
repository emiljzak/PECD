from mayavi import mlab

import numpy as np
from scipy.special import sph_harm
from scipy import interpolate

import os

import json

import input
import GRID
import MAPPING
import CONSTANTS

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
            c.append(complex(words[5+ivec]))
        coeffs.append([i,n,xi,l,m,np.asarray(c)])
    return coeffs


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

    else:

        val = f(i,n,r,Gr,nlobs,nbins) * np.sqrt( float( w[n] ) ) **(-1) 
        #print(type(val),np.shape(val))
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


def plot_snapshot_int(params,psi,maparray,Gr,t,flist, irun):
    #plot snapshots using interpolation for the chi functions
    #make it general
    nlobs = params['bound_nlobs']
    nbins = params['bound_nbins'] 
    npoints = 360
    rmax    = nbins * params['bound_binw']

    #fig = plt.figure(figsize = (3.,3.), dpi=200, constrained_layout=True)
    #spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    #ax_radang_r = fig.add_subplot(spec[0, 0],projection='polar')
    #if params['plot_types']['r-radial_angular'] == True:
    """================ radial-angular in real space ==============="""
    width = 0.0
    for elem in params['FEMLIST']:
        width += elem[0] * elem[2]

    plot_wf_angrad_int_XZ(0.0, rmax, npoints, nlobs, nbins, psi, maparray, Gr, params, t, flist, irun)
    plot_wf_angrad_int_XY(0.0, rmax, npoints, nlobs, nbins, psi, maparray, Gr, params, t, flist, irun)

    
def interpolate_chi(Gr,nlobs,nbins,binw,maparray):

    w       =  np.zeros(nlobs)
    xx,w    =  GRID.gauss_lobatto(nlobs,14)
    w       =  np.array(w)

    interpolation_step = binw/200.0
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
    plt.close()

def plot_wf_angrad_int_XZ(rmin,rmax,npoints,nlobs,nbins,psi,maparray,Gr,params,t,flist,irun):
    #================ radial-angular in real space ===============#

    coeff_thr = 1e-6
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

    phi0 = 0.0 * np.pi/2 #fixed polar angle. #YZ plane
    #here we can do phi-averaging
    
    for ielem, elem in enumerate(maparray):
        if np.abs(psi[ielem]) > coeff_thr:
            #print(str(elem) + str(psi[ielem]))

            chir = flist[elem[2]-1](rang) #labelled by xi

            for i in range(len(rang)):
                y[i,:] +=  psi[ielem]  * spharm(elem[3], elem[4], gridtheta1d[i], phi0) * \
                           chir #chi(elem[0], elem[1], rang[:], Gr, w, nlobs, nbins) 

    line_angrad_r = axradang_r.contourf(thetamesh, rmesh, np.abs(y)/np.max(np.abs(y)), 
                                        ncontours, cmap = 'jet', vmin=0.0, vmax=1.0) #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    #plt.colorbar(line_angrad_r, ax=axradang_r, aspect=30)
    axradang_r.set_rlabel_position(100)
    #axradang_r.set_yticklabels(list(str(np.linspace(rmin,rmax,5.0)))) # set radial tick label
    #plt.legend()   
    plt.show()  
    if params["save_snapthots"] == True:

        if params['field_type']['function_name'] == "fieldCPL":
            if params['field_type']['typef'] == "LCPL":
                helicity = "L"
            elif params['field_type']['typef'] == "RCPL":
                helicity = "R"
        else:
            helicity = "0"
        fig.savefig( params['job_directory']  + "/animation/" + helicity + "_angrad_YZ" + "_" + str(irun) + "_t=" +\
                     str("%4.1f"%(t/np.float64(1.0/24.188)))+"_.png" ,\
                     bbox_inches='tight')
    plt.close()

def plot_wf_angrad_int_XY(rmin,rmax,npoints,nlobs,nbins,psi,maparray,Gr,params,t,flist,irun):
    #================ radial-angular in real space ===============#

    coeff_thr = 1e-6
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
                                        ncontours, cmap = 'jet', vmin=0.0, vmax=1.0) #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    #plt.colorbar(line_angrad_r, ax=axradang_r, aspect=30)
    axradang_r.set_rlabel_position(100)
    #axradang_r.set_yticklabels(list(str(np.linspace(rmin,rmax,5.0)))) # set radial tick label
    #plt.legend()   
    #plt.show()  
    if params["save_snapthots"] == True:
        if params['field_type']['function_name'] == "fieldCPL":
            if params['field_type']['typef'] == "LCPL":
                helicity = "L"
            elif params['field_type']['typef'] == "RCPL":
                helicity = "R"
        else:
            helicity = "0"
        fig.savefig( params['job_directory']  + "/animation/" + helicity + "_angrad_XY" + "_" + str(irun) + "_t=" +\
                     str("%4.1f"%(t/np.float64(1.0/24.188)))+"_.png" ,\
                     bbox_inches='tight')
    plt.close()

def plot_initial_orbitals(params,maparray,orbitals):
    maparray = np.asarray(maparray)

    nlobs = params['bound_nlobs']
    nbins = params['bound_nbins'] 
    npoints = 360
    rmax    = nbins * params['bound_binw']
    rmin = 0.0
    Gr_all, Nr_all = GRID.r_grid_prim( params['bound_nlobs'], nbins , params['bound_binw'],  params['bound_rshift'] )

    maparray_chi, Nbas_chi = MAPPING.GENMAP_FEMLIST( params['FEMLIST'],  0, \
                                params['map_type'], params['job_directory'] )

    flist = interpolate_chi(Gr_all, params['bound_nlobs'], nbins, params['bound_binw'], maparray_chi)
    
    for iorb in range(params['ivec']+2):
        psi = orbitals[:,iorb]
        plot_iorb(rmin,rmax,npoints,nlobs,nbins,psi,maparray,params,flist,iorb)


def plot_iorb(rmin,rmax,npoints,nlobs,nbins,psi,maparray,params,flist,iorb):
    #================ radial-angular in real space ===============#

    coeff_thr = 1e-5
    ncontours = 100

    fig_XY = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec_XY = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_XY)
    axradang_r_XY = fig_XY.add_subplot(spec_XY[0, 0], projection='polar')

    rang = np.linspace(rmin, rmax, npoints, endpoint=True, dtype=float)
    unity_vec = np.linspace(0.0, 1.0, npoints, endpoint=True, dtype=float)
    gridphi1d = 2 * np.pi * unity_vec

    rmesh, phimesh = np.meshgrid(rang, gridphi1d)

    fig_XZ = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec_XZ = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_XZ)
    axradang_r_XZ = fig_XZ.add_subplot(spec_XZ[0, 0], projection='polar')

    rang = np.linspace(rmin, rmax, npoints, endpoint=True, dtype=float)
    unity_vec = np.linspace(0.0, 1.0, npoints, endpoint=True, dtype=float)
    gridtheta1d = 2 * np.pi * unity_vec

    rmesh, thetamesh = np.meshgrid(rang, gridtheta1d)

    x   =  np.zeros(nlobs)
    w   =  np.zeros(nlobs)
    x,w =  GRID.gauss_lobatto(nlobs,14)
    w   =  np.array(w)
    x   =  np.array(x) # convert back to np arrays

    y_XY = np.zeros((len(rang),len(rang)), dtype=complex)
    y_XZ = np.zeros((len(rang),len(rang)), dtype=complex)

    theta0  = np.pi/2 #fixed azimuthal angle
    phi0    = 0.0 * np.pi/2 #fixed polar angle. #YZ plane
 
    for ielem, elem in enumerate(maparray):
        if np.abs(psi[ielem]) > coeff_thr:
            #print(str(elem) + str(psi[ielem]))

            chir = flist[elem[2]-1](rang) #labelled by xi

            for i in range(len(rang)):
                y_XY[i,:] +=  psi[ielem]  * spharm(elem[3], elem[4], theta0, gridphi1d[i]) * \
                           chir #chi(elem[0], elem[1], rang[:], Gr, w, nlobs, nbins) 
                y_XZ[i,:] +=  psi[ielem]  * spharm(elem[3], elem[4], gridtheta1d[i], phi0) * \
                           chir #chi(elem[0], elem[1], rang[:], Gr, w, nlobs, nbins) 


    line_angrad_r_XY = axradang_r_XY.contourf(phimesh, rmesh, np.abs(y_XY)/np.max(np.abs(y_XY)), 
                                        ncontours, cmap = 'jet', vmin=0.0, vmax=0.5) #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    
    line_angrad_r_XZ = axradang_r_XZ.contourf(thetamesh, rmesh, np.abs(y_XZ)/np.max(np.abs(y_XZ)), 
                                        ncontours, cmap = 'jet', vmin=0.0, vmax=0.5) #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    #plt.colorbar(line_angrad_r, ax=axradang_r, aspect=30)
    axradang_r_XY.set_rlabel_position(100)
    axradang_r_XZ.set_rlabel_position(100)
    #axradang_r.set_yticklabels(list(str(np.linspace(rmin,rmax,5.0)))) # set radial tick label
    #plt.legend()   
    #plt.show()  
    if params["save_snapthots"] == True:
        if params['field_type']['function_name'] == "fieldCPL":
            if params['field_type']['typef'] == "LCPL":
                helicity = "L"
            elif params['field_type']['typef'] == "RCPL":
                helicity = "R"
        else:
            helicity = "0"

        fig_XY.savefig( params['job_directory']  + "/animation/" + helicity + "_angrad_XY" + \
                        "_" + "orb_" + str(iorb) +"_.png" ,\
                        bbox_inches='tight')

        fig_XZ.savefig( params['job_directory']  + "/animation/" + helicity + "_angrad_YZ" + \
                        "_" + "orb_" + str(iorb) + "_.png" ,\
                        bbox_inches='tight')
    plt.close()


def plot_snapshot(params,psi,maparray,Gr,t):
    #make it general
    nlobs = params['bound_nlobs']
    nbins = params['bound_nbins'] 
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


def plot_elfield(Fvec,tgrid,time_to_au):
    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("time (as)")
    plt.ylabel("normalized Field components")
    ax.scatter(tgrid/time_to_au, -Fvec[0].real, label = "Field-x", marker = '.', color = 'r', s = 1)
    ax.scatter(tgrid/time_to_au, Fvec[0].imag, label = "Field-y", marker = '.', color = 'g', s = 1)
    ax.scatter(tgrid/time_to_au, Fvec[1], label = "Field-z", marker = '.', color = 'b', s = 1)
    ax.legend()
    plt.show()

def plot_2D_polar_map(func,grid_theta,kgrid,ncontours,params):
    fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    axft = fig.add_subplot(spec[0, 0], projection='polar')
    kmesh, thetamesh = np.meshgrid(kgrid,grid_theta)
    axft.set_ylim(0,2.5) #radial extent
    line_ft = axft.contourf(thetamesh, kmesh, func/np.max(func), 
                            ncontours, cmap = 'jet') #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    plt.colorbar(line_ft, ax=axft, aspect=30)
    plt.legend()   
    #fig.savefig( params['job_directory']  + "/MF-MPAD.png",bbox_inches='tight')

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


def plot_rotdens(rotdens, grid):
    # Create a sphere
    r = 0.3
    theta = grid[:,0]
    chi = grid[:,1]

    #theta, chi = np.meshgrid(theta1d,chi1d)

    print(theta.shape)
    x =  np.sin(theta) * np.cos(chi)
    y =  np.sin(theta) * np.sin(chi)
    z =  np.cos(theta)

    s = rotdens
    #mlab.mesh(x - m, y - n, z, scalars=s, colormap='jet')
    
    s /= s.max()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    plot = ax.plot_trisurf(
        x, y, s ,cmap=plt.cm.Spectral )

    plt.show()
    exit()
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
    mlab.clf()
    # Represent spherical harmonics on the surface of the sphere
    s = rotdens
    #mlab.mesh(x - m, y - n, z, scalars=s, colormap='jet')
    
    s /= s.max()
    mlab.points3d(x, y, z, s)
    #mlab.mesh(s * x, s * y , s * z + 1.3,
    #        scalars=s, colormap='Spectral')

    #mlab.view(90, 70, 6.2, (-1.3, -2.9, 0.25))
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
    
    ivec = 0

    x, y, z = np.mgrid[xmin:xmax:npts, ymin:ymax:npts, zmin:zmax:npts]
    #wf = np.sin(x**2 + y**2 + 2. * z**2)
    wf = calc_wf_xyzgrid(nlobs,nbins,ivec,Gr,wffile,[x,y,z])
    fmin = wf.min()
    fmax = wf.max()

    wf2 = wf.reshape(int(np.abs(npts)),int(np.abs(npts)),int(np.abs(npts)))

    #plot volume
    mlab.pipeline.volume(mlab.pipeline.scalar_field(wf2),  vmin=fmin + 0.25 * (fmax - fmin),
                                   vmax=fmin + 0.9 * (fmax - fmin))
    

    mlab.view(132, 54, 45, [21, 20, 21.5])  
    mlab.show()


def build_cube(params,Gr,wffile):
    
    ivec_min = 0
    ivec_max = 7

    npts = 50j
    npt = 50
    cube_range = 5.0
    dx = 2*cube_range/npt
    dy = dx
    dz = dx

    xmax = cube_range
    xmin = -1.0 * cube_range
    
    zmax = cube_range
    zmin = -1.0 * cube_range
    
    ymax = cube_range
    ymin = -1.0 * cube_range
    
    x1d = np.linspace(xmin,xmax,npt)

    
    x, y, z = np.mgrid[xmin:xmax:npts, ymin:ymax:npts, zmin:zmax:npts]
    #x, y, z = np.meshgrid(x1d, x1d, x1d)
    print(np.shape(x))
    #wf = np.sin(x**2 + y**2 + 2. * z**2)
    
    print(params['job_directory'])
    for ivec in range(ivec_min,ivec_max+1):

        wf = calc_wf_xyzgrid(params['bound_nlobs'],params['bound_nbins'],ivec,Gr,wffile,[x,y,z])

        wf_cube = wf.reshape(npt,npt,npt)
    
        cubefile = open(params['job_directory'] + "/" + str(ivec)+".cub", 'w')

        """ print header"""
        cubefile.write( "CO orbitals" + "\n" + "normalized electronic density" + "\n")
        cubefile.write( "2  " + str(-1.0*cube_range) + " " + str(-1.0*cube_range) + " "+ str(-1.0*cube_range) + "\n" +\
            str(npt) + " " + str(dx) + "  0.000000    0.000000" + "\n" +\
            str(npt) + " " +  "0.000000 " +  str(dy) + " 0.000000" + "\n" +\
            str(npt) + " " +  "0.000000    0.000000 " +  str(dz)  + "\n" +\
            " 6    6.000000    0.000000    0.000000    -1.230854" + "\n" +\
            " 8    8.000000    0.000000    0.000000     0.923434" + "\n" )

        for ix in range(npt):
            for iy in range(npt):
                for iz in range(npt):
                    #cubefile.write( str(x[ix,iy,iz]) + " "+ str(y[ix,iy,iz]) + " "+ str(z[ix,iy,iz]) + " "+str(wf_cube[ix,iy,iz]))
                    cubefile.write( "%12.6e"%wf_cube[ix,iy,iz] + " ")#
                    if iz%6==5:
                        cubefile.write("\n")
                cubefile.write("\n")
    return wf_cube

def calc_wf_xyzgrid(nlobs,nbins,ivec,Gr,wffile,grid):

    coeffs = read_coeffs(wffile,ivec+1)

    xl      =   np.zeros(nlobs)
    w       =   np.zeros(nlobs)
    xl,w    =   GRID.gauss_lobatto(nlobs,14)
    w       =   np.array(w)

    X, Y, Z = grid[0], grid[1], grid[2]
    #print(X.shape)
    val = np.zeros((X.shape[0] * X.shape[0] * X.shape[0]), dtype = complex)

   # chilist = PLOTS.interpolate_chi(    Gr_prim, 
   #                                     params['bound_nlobs'], 
   #                                     params['bound_nbins'], 
   #                                     params['bound_binw'], 
    #                                    maparray_chi)


    rx,thetax,phix  = cart2sph(X,Y,Z)
    r               = rx.flatten()
    theta           = thetax.flatten()
    phi             = phix.flatten()

    #print(theta.shape)
    #print(np.shape(coeffs))
    #print(type(coeffs))


    for icount, ipoint in enumerate(coeffs):
        print(icount)
        #print(ipoint[5][ivec])
        if np.abs(ipoint[5][ivec]) > 1e-3:
            val +=  ipoint[5][ivec] * chi(ipoint[0], ipoint[1],r,Gr,w,nlobs,nbins) * spharm(ipoint[3], ipoint[4], theta, phi)

    #val *= np.sin(theta) 
    #val.real/(np.max(val))#
    return  np.abs(val)**2/ (np.max(np.abs(val)**2))


def calc_wf_xyzgrid_int(params,ivec,Gr,wffile,grid):

    nlobs = params['bound_nlobs']
    nbins = params['bound_nbins']
    binw = params['bound_binw']
    
    coeffs = read_coeffs(wffile,ivec+1)

    xl      =   np.zeros(nlobs)
    w       =   np.zeros(nlobs)
    xl,w    =   GRID.gauss_lobatto(nlobs,14)
    w       =   np.array(w)

    X, Y, Z = grid[0], grid[1], grid[2]
    #print(X.shape)
    val = np.zeros((X.shape[0] * X.shape[0] * X.shape[0]), dtype = complex)

    chilist = PLOTS.interpolate_chi(    Gr_prim, 
                                        params['bound_nlobs'], 
                                        params['bound_nbins'], 
                                        params['bound_binw'], 
                                        maparray_chi)


    rx,thetax,phix  = cart2sph(X,Y,Z)
    r               = rx.flatten()
    theta           = thetax.flatten()
    phi             = phix.flatten()

    #print(theta.shape)
    #print(np.shape(coeffs))
    #print(type(coeffs))


    for icount, ipoint in enumerate(coeffs):
        print(icount)
        #print(ipoint[5][ivec])
        if np.abs(ipoint[5][ivec]) > 1e-3:
            val +=  ipoint[5][ivec] * chi(ipoint[0], ipoint[1],r,Gr,w,nlobs,nbins) * spharm(ipoint[3], ipoint[4], theta, phi)

    #val *= np.sin(theta) 
    #val.real/(np.max(val))#
    return  np.abs(val)**2/ (np.max(np.abs(val)**2))

def find_nearest(array, value):
    array   = np.asarray(array)
    idx     = (np.abs(array - value)).argmin()
    return array[idx], idx

def plot_pad_polar(params,klist,helicity):
    """ polar plot of angular distribution of photoelectron momentum for a given energy (wavevector)"""

    with open( params['job_directory']+ "grid_W_av" , 'r') as gridfile:   
        grid = np.loadtxt(gridfile)

    with open( params['job_directory'] + "W" + "_"+ str(helicity) + "_av_3D_0", 'r') as Wavfile:   
        Wav = np.loadtxt(Wavfile)

    fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    ind_kgrid   = [] #index of electron momentum in the list
    ax = fig.add_subplot(projection="polar", facecolor="lightgoldenrodyellow")
    plt.legend(loc="lower left",bbox_to_anchor=(1.5,1.0))
    ax.tick_params(grid_color="palegoldenrod")
    ax.set_rlabel_position(70)
    #ax.set_yticklabels(list(str(np.linspace(rmin,rmax,5.0)))) # set radial tick label

    for kelem in klist:
        k, ind = find_nearest(grid[0], kelem)
        ind_kgrid.append(ind)
        
    #find maximum
    Wmax = np.max(Wav[:,ind_kgrid])
    for i,kelem in enumerate(klist):
        k, ind = find_nearest(grid[0], kelem)
        ax.plot(grid[1], Wav[:,ind_kgrid[i]]/Wmax,label=str("%5.1f"%(0.5*k**2 * CONSTANTS.au_to_ev)))

    thetagrid = np.linspace(0,2.0*np.pi,400)


    plt.legend()   
    fig.savefig( params['job_directory']  + "/MF-PAD.png",bbox_inches='tight')
    #plt.show()  
    plt.close()




def plot_wf_isosurf(nlobs,nbins,Gr,wffile):
    
    ivec = 2
    mlab.clf()
    fig = mlab.figure(1, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(600, 600))

    plot_molecule = False

    if plot_molecule == True:
        # The position of the atoms
        scale = 1.0#40.0 / 5.5
        trans = np.array([0.0, 0.0, 0.0])
        atoms_O = np.array([0.0, -0.02, 0.21]) * scale + trans
        atoms_H1 = np.array([0.0, -1.83,   -1.53,  ]) * scale + trans
        atoms_H2 = np.array([ 0.0,    2.15,    -1.89]) * scale + trans



        O = mlab.points3d(atoms_O[0], atoms_O[1], atoms_O[2],
                        scale_factor=3,
                        resolution=20,
                        color=(1, 1, 0.2),
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

    npts = 50j
    grange = 15.0
    
    xmax = grange
    xmin = -1.0 * grange
    
    zmax = grange
    zmin = -1.0 * grange
    
    ymax = grange
    ymin = -1.0 * grange
    

    x, y, z = np.mgrid[xmin:xmax:npts, ymin:ymax:npts, zmin:zmax:npts]
    #wf = np.sin(x**2 + y**2 + 2. * z**2)
    wf = calc_wf_xyzgrid(nlobs,nbins,ivec,Gr,wffile,[x,y,z])
    fmin = wf.min()
    fmax = wf.max()
    #print(wf)
    wf2 = wf.reshape(int(np.abs(npts)),int(np.abs(npts)),int(np.abs(npts)))

    #plot isosurface
    mywf = mlab.contour3d(wf2, contours=[0.1,0.2,0.3,0.5,0.9], colormap='gnuplot',opacity=0.5) #[0.9,0.7,0.5,0.4]

    mlab.show()




if __name__ == "__main__":      

    # preparation of parameters
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


    print(" ")
    print("---------------------- PLOTS --------------------")
    print(" ")

    os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

    
    directory = "/home/emil/Desktop/projects/PECD/tests/molecules/chiralium/chiralium_4_10_2.00_150_R"
    os.chdir(directory)
    
    path = os.getcwd()
    print("dir:" + path)

    with open('input', 'r') as input_file:
        params = json.load(input_file)

    
    #plot_pad_polar(params['k_list_pad'],0)

    wffile = params['job_directory'] + params['file_psi_init']+"_0"  # "psi0_h2o_1_12_30.0_4_uhf_631Gss.dat"# "psi_init_h2o_3_24_20.0_2_uhf_631Gss.dat"#+ "psi0_h2o_1_20_10.0_4_uhf_631Gss.dat"
    nvecs = 10 #how many vectors to load?
    ivec = params['ivec'] #which vector to plot?
    
    coeffs = read_coeffs(wffile,nvecs)

    #psi = np.zeros(len(coeffs), dtype = complex)
    #print(psi.shape)
    #for ielem,elem in enumerate(coeffs):
    #    psi[ielem] = elem[5][ivec]

    #Gr, Nr = GRID.r_grid( params['bound_nlobs'], nbins , params['bound_binw'],  params['bound_rshift'] )
    Gr, Nr = GRID.r_grid_prim( params['bound_nlobs'], params['bound_nbins']  , params['bound_binw'],  params['bound_rshift'] )
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
    #r0 = 2.0
    #coeffs = 0.0
    #Gr   = 0.0
    #plot_wf_ang(r0,coeffs,Gr,params['bound_nlobs'], params['bound_nbins'])

    """ plot angular-radial wavefunction on a polar plot"""
    #plot_snapshot(params,psi,maparray,Gr,t)
    #plot_wf_angrad( 0.0, params['bound_binw'] * params['bound_nbins'], 200,params['bound_nlobs'], nbins,\
    #                psi,maparray ,Gr, params, 0.0)
    #plot_wf_angrad(rmin,rmax,npoints,nlobs,nbins,psi,maparray,rgrid,params,t)

    """ plot 3D isosurface of the wavefunction amplitude (density)"""
    plot_wf_isosurf(params['bound_nlobs'], params['bound_nbins'], Gr, wffile)


    """ plot 4D volume of the wavefunction amplitude (density)"""
    #plot_wf_volume(params['bound_nlobs'], params['bound_nbins'],Gr,wffile)


    """ generate cube file"""
    #build_cube(params,Gr,wffile)
    exit()

