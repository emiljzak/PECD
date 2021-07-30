import numpy as np

import quaternionic
import spherical
import itertools
import os
import sys
import matplotlib.pyplot as plt
from scipy.special import eval_legendre
from scipy import integrate
from scipy import interpolate
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.gridspec as gridspec

import PLOTS
import CONSTANTS

def gen_euler_grid(n_euler):
    alpha_1d        = list(np.linspace(0, 2*np.pi,  num=n_euler, endpoint=False))
    beta_1d         = list(np.linspace(0, np.pi,    num=n_euler, endpoint=True))
    gamma_1d        = list(np.linspace(0, 2*np.pi,  num=n_euler, endpoint=False))
    euler_grid_3d   = np.array(list(itertools.product(*[alpha_1d, beta_1d, gamma_1d]))) #cartesian product of [alpha,beta,gamma]

    #we can choose meshgrid instead
    #euler_grid_3d_mesh = np.meshgrid(alpha_1d, beta_1d, gamma_1d)
    #print(euler_grid_3d_mesh[0].shape)
    #print(np.vstack(euler_grid_3d_mesh).reshape(3,-1).T)
    #print(euler_grid_3d)
    #exit()
    n_euler_3d      = euler_grid_3d.shape[0]
    print("\nTotal number of 3D-Euler grid points: ", n_euler_3d , " and the shape of the 3D grid array is:    ", euler_grid_3d.shape)
    #print(euler_grid_3d)
    return euler_grid_3d, n_euler_3d

def gen_spherical_grid(n_euler):
    phi_1d        = list(np.linspace(0, 2*np.pi,  num=n_euler, endpoint=False))
    theta_1d         = list(np.linspace(0, np.pi,    num=n_euler, endpoint=True))

    euler_grid_2d   = np.array(list(itertools.product(*[theta_1d, phi_1d]))) #cartesian product of [alpha,beta,gamma]

    #we can choose meshgrid instead
    #euler_grid_3d_mesh = np.meshgrid(alpha_1d, beta_1d, gamma_1d)
    #print(euler_grid_3d_mesh[0].shape)
    #print(np.vstack(euler_grid_3d_mesh).reshape(3,-1).T)
    #print(euler_grid_3d)
    #exit()
    n_euler_2d      = euler_grid_2d.shape[0]
    print("\nTotal number of 3D-Euler grid points: ", n_euler_2d , " and the shape of the 3D grid array is:    ", euler_grid_2d.shape)
    #print(euler_grid_3d)
    print(euler_grid_2d.shape)
    euler_grid_3d = np.zeros((euler_grid_2d.shape[0],3))
    euler_grid_3d[:,:-1] = euler_grid_2d
    print(euler_grid_3d.shape)
    return euler_grid_3d, n_euler_2d

def rotate_spharm(func,D):
    rot_angle = [0.0, np.pi/4, 0.0]
    func_rotated = np.zeros((func.shape[0], func.shape[1]))
    for m in range(D.shape[1]):
        func_rotated += D[:,m] * func[:,m]

    return func_rotated

def test_wigner():
    """ Test wigner functions: orthogonality, symmetry and plots (2D slices of known functions). Decide how to store them """

    Jmax = 2
    wigner = spherical.Wigner(Jmax)

    """
    #grid
    alpha = 0.4 * np.pi
    beta = 0.5 * np.pi 
    gamma = 0.2 * np.pi

    ngrid = 1

    WDMATS = []

    R = quaternionic.array.from_euler_angles(alpha, beta, gamma)
    D = wigner.D(R)
    print(wigner.Dindex(1,0,-1))
    print("D:")
    print(D)
    for J in range(Jmax+1):
        WDM = np.zeros((2*J+1,2*J+1,ngrid), dtype=complex)
        for m in range(-J,J+1):
            for k in range(-J,J+1):
                WDM[m+J,k+J,:] = D[wigner.Dindex(J,m,k)]
        print(J,WDM)
        print(WDM.shape)

        WDMATS.append(WDM)        
    print("WDMATS:")
    print(WDMATS)
    
    
    #test orthogonality
    J = 4
    SMAT = np.zeros((2*J+1,2*J+1), dtype = complex)
    WDM = WDMATS[J]
    for k1 in range(-J,J+1):
        for k2 in range(-J,J+1):
            SMAT[k1,k2] = np.sum(WDM[:,k1+J,0] * np.conjugate(WDM[:,k2+J,0]) )
    print("SMAT: ")
    with np.printoptions(precision=4, suppress=True, formatter={'complex': '{:15.8f}'.format}, linewidth=400):
        print(SMAT)
    """

    """
    #plot D_m0^J(theta,phi)
    nphi = 181
    ntheta = 91
    theta_1d = np.linspace(0,   np.pi,  2*ntheta) # 
    phi_1d   = np.linspace(0, 2*np.pi, 2*nphi) # 
    grid = np.meshgrid(phi_1d, theta_1d, phi_1d)
    N_Euler = 50
    #calculate D matrix on the grid
    grid_euler, n_grid_euler = gen_spherical_grid(N_Euler)
    R = quaternionic.array.from_euler_angles(grid_euler)

    D = wigner.D(R)
    J = 1
    m = 0
    k = 0
    WDM = np.zeros((R.shape[0]), dtype=complex)
    WDM = D[:,wigner.Dindex(J,m,k)]
    
    #PLOTS.plot_3D_angfunc(WDM,grid_euler)
    """

    # plot rotated spherical harmonics
    alpha = 0.0 * np.pi
    beta = 0.5 * np.pi 
    gamma = 0.0 * np.pi

    ngrid = 1
    R = quaternionic.array.from_euler_angles(alpha, beta, gamma)
    D = wigner.D(R)

    for J in range(Jmax+1):
        WDM = np.zeros((2*J+1,2*J+1,ngrid), dtype=complex)
        for m in range(-J,J+1):
            for k in range(-J,J+1):
                WDM[m+J,k+J,:] = D[wigner.Dindex(J,m,k)]

    PLOTS.plot_spharm_rotated(WDM[:,:,0])


def legendre_expansion(grid,Wav,Lmax):
    """ Calculate Legendre expansion coefficients as a function of photo-electron momentum """
    """ Calculate photo-electron energy spectrum """

    kgrid       = grid[0]
    thetagrid   = grid[1]

    """ Interpolate W(k,theta)"""
    W_interp    = interpolate.interp2d(kgrid, thetagrid, Wav, kind='cubic')

    #fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    #spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    #ax = fig.add_subplot(spec[0, 0], projection='polar')
    #ax.set_ylim(0,1) #radial extent

    #plot W_av on the original grid
    #W_interp_mesh = W_interp(kgrid, thetagrid)
    #kmesh,thetamesh  = np.meshgrid(kgrid, thetagrid)
    #line_W_original = ax.contourf(thetamesh, kmesh, W_interp_mesh, 
    #                        100, cmap = 'jet') 
    #plt.show()

    #plot W_av on test grid 
    #thetagridtest       = np.linspace(-np.pi, np.pi, 200)
    #kgridtest           = np.linspace(0, 1, 200)
    #kgridtestmesh ,thetatestmesh    = np.meshgrid(kgridtest, thetagridtest )

    #W_interp_testmesh   = W_interp( kgridtest , thetagridtest )
    #line_test = ax.contourf(thetatestmesh, kgridtestmesh  , W_interp_testmesh , 
    #                        100, cmap = 'jet') 
    #plt.show()

    # Define function and interval
    deg     = Lmax + 10
    nleg    = deg
    x, w    = np.polynomial.legendre.leggauss(deg)
    w       = w.reshape(nleg,-1)

    nkpoints    = 100
    bcoeff      = np.zeros((nkpoints,Lmax+1), dtype = float)
    spectrum    = np.zeros(nkpoints, dtype = float)
    kgrid       = np.linspace(0.05,1.0,nkpoints)

    """ calculating Legendre moments """
    for n in range(0,Lmax+1):
        Pn = eval_legendre(n, x).reshape(nleg,-1)
        for ipoint,k in enumerate(list(kgrid)):
            W_interp1 = W_interp(k,-np.arccos(x)).reshape(nleg,-1)
            print(k)
            bcoeff[ipoint,n] = np.sum(w[:,0] * W_interp1[:,0] * Pn[:,0]) * (2.0 * n + 1.0) / 2.0
        #plt.plot(kgrid,bcoeff[:,n],label=n)
        #plt.legend()   
    #plt.show()
   
    """ calculating photo-electron spectrum """
    for ipoint,k in enumerate(list(kgrid)):   
        W_interp1 = W_interp(k,-np.arccos(x)).reshape(nleg,-1) 
        spectrum[ipoint] = np.sum(w[:,0] * W_interp1[:,0] * np.sin(np.arccos(x)) )
    #plt.plot(kgrid,spectrum/spectrum.max(), label = r"$\sigma(k)$", marker = '.', color = 'r')
    plt.plot((0.5*kgrid**2)*CONSTANTS.au_to_ev,spectrum/spectrum.max(), label = r"$\sigma(k)$", marker = '.', color = 'r')
    plt.xlabel("Energy (eV)")
    plt.xlim([0,6]) 
   #plt.xlabel("momentum (a.u.)")
    plt.ylabel("cross section")
    plt.legend()   
    plt.show()
    exit()

    #plot Legendre-reconstructed W_av on test grid 
    thetagridtest       = np.linspace(-np.pi,np.pi, nkpoints)
    kgridtest           = np.linspace(0.05, 1, nkpoints)
    kgridtestmesh, thetatestmesh    = np.meshgrid(kgridtest, thetagridtest )

    #W_interp_testmesh   = W_interp( kgridtest , thetagridtest )

    W_legendre = np.zeros((kgridtest.shape[0],thetagridtest.shape[0]), dtype = float)

    for ipoint in range(nkpoints):

        for n in range(0,Lmax+1):
            Pn = eval_legendre(n, np.cos(thetagridtest)).reshape(thetagridtest.shape[0],1)
            W_legendre[ipoint,:] += bcoeff[ipoint,n] * Pn[:,0] 

    
    fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax = fig.add_subplot(spec[0, 0], projection='polar')
    ax.set_ylim(0,1) #radial extent
    line_legendre = ax.contourf(   thetatestmesh ,kgridtestmesh, W_legendre.T / W_legendre.max(), 100, cmap = 'jet') 
    plt.colorbar(line_legendre, ax=ax, aspect=30) 
    #plt.show()

    return bcoeff, kgrid, spectrum/spectrum.max()


def analyze_Wav(N_batches,params):

    with open( params['job_directory'] + "grid_W_av" , 'r') as gridfile:   
        grid = np.loadtxt(gridfile)

    if params['field_type']['typef'] == "LCPL":
        helicity = "L"
    elif params['field_type']['typef'] == "RCPL":
        helicity = "R"
    else:
        helicity = "0"


    grid = grid.T
    Wav = np.zeros((grid.shape[0],grid.shape[0]), dtype = float)  
    Wavi = np.zeros((grid.shape[0],grid.shape[0]), dtype = float)
    grid = grid.T

    batch_list = [0]
    for icount, ibatch in enumerate(batch_list):
        with open( params['job_directory']+ "W" + "_"+ helicity + "_av_3D_" + str(ibatch), 'r') as Wavfile:   
            Wavi = np.loadtxt(Wavfile)
        #PLOTS.plot_2D_polar_map(Wavi,grid[1],grid[0],100)
        Wav += Wavi

    with open( params['job_directory']  +  "W" + "_"+ helicity + "_av_3D" , 'w') as Wavfile:   
        np.savetxt(Wavfile, Wav, fmt = '%10.4e')


    #PLOTS.plot_2D_polar_map(Wav,grid[1],grid[0],100)

    return grid, Wav

def find_nearest(array, value):
    array   = np.asarray(array)
    idx     = (np.abs(array - value)).argmin()
    return array[idx], idx





def calc_pecd(N_batches,params,bcoeff,kgrid):
    
    with open( params['job_directory']+ "grid_W_av" , 'r') as gridfile:   
        grid = np.loadtxt(gridfile)

    with open( params['job_directory'] + "W_R_av_3D", 'r') as Wavfile:   
        WavR = np.loadtxt(Wavfile)

    with open( params['job_directory'] + "W_L_av_3D", 'r') as Wavfile:   
        WavL = np.loadtxt(Wavfile)


    print("Quantitative PECD analysis")
    k_pecd      = [] #values of electron momentum at which PECD is evaluated
    ind_kgrid   = [] #index of electron momentum in the list

    for kelem in params['k_pecd']:
        k, ind = find_nearest(kgrid, kelem)
        k_pecd.append(k)
        ind_kgrid.append(ind)


    pecd_sph = []
    pecd_mph = []

    for ielem, k in enumerate(k_pecd):
        print(str('{:8.2f}'.format(k)) + " ".join('{:12.8f}'.format(bcoeff[ielem,n]/bcoeff[ielem,0]) for n in range(params['pecd_lmax']+1)) + "\n")
        pecd_sph.append(2.0 * bcoeff[ielem,1]/bcoeff[ielem,0] * 100.0)

    pecd_pad = (WavR - WavL)#/(np.abs(WavR)+np.abs(WavL)+1.0) #(WavL+WavR)  #/ 
    print("plotting PECD")
    PLOTS.plot_2D_polar_map(pecd_pad,grid[1],grid[0],100)

    return pecd_sph


if __name__ == "__main__": 
    os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

    jobtype = "local"
    inputfile = "input_d2s"

    os.environ['KMP_DUPLICATE_LIB_OK']= 'True'
    import importlib
    input_module = importlib.import_module(inputfile)
    print("importing input file module: " + inputfile)
    print("jobtype: " + str(jobtype))
    print(" ")
    print("---------------------- INPUT ECHOs --------------------")
    print(" ")
    params = input_module.gen_input(jobtype) 


    N_batches = 1

    grid, Wav = analyze_Wav(N_batches,params) #return fully averaged (over batches) PAD

    bcoeff, kgrid, spectrum = legendre_expansion(grid,Wav,params['pecd_lmax']) #returns legendre expansion coefficients of PAD

    pecd_sph = calc_pecd(N_batches,params,bcoeff,kgrid) #calculates PECD and other related quantities

    print("Single-photon contribution to PECD: " + str(pecd_sph) +"%" )
