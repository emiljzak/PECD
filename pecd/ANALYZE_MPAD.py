import numpy as np

import quaternionic
import spherical
import itertools
import os
import sys

import PLOTS

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


def legendre_expansion(Wav,grid,Lmax):
    """ Calculate Legendre expansion coefficients as a function of photo-electron momentum """

    bcoeffs = np.zeros(grid.shape[0])

    return bcoeffs


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
        PLOTS.plot_2D_polar_map(Wavi,grid[1],grid[0],100)
        Wav += Wavi

    with open( params['job_directory']  +  "W" + "_"+ helicity + "_av_3D" , 'w') as Wavfile:   
        np.savetxt(Wavfile, Wav, fmt = '%10.4e')


    PLOTS.plot_2D_polar_map(Wav,grid[1],grid[0],100)



def calc_pecd(N_batches,params):
    
    with open( params['job_directory']+ "grid_W_av" , 'r') as gridfile:   
        grid = np.loadtxt(gridfile)

    with open( params['job_directory'] + "W_R_av_3D", 'r') as Wavfile:   
        WavR = np.loadtxt(Wavfile)

    with open( params['job_directory'] + "W_L_av_3D", 'r') as Wavfile:   
        WavL = np.loadtxt(Wavfile)


    pecd = WavR-WavL #/ (np.abs(WavR)+np.abs(WavL))
    print("plotting PECD")
    PLOTS.plot_2D_polar_map(pecd,grid[1],grid[0],100)


    return pecd


if __name__ == "__main__": 

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
    #analyze_Wav(N_batches,params)
    calc_pecd(N_batches,params) 

    #analyze_Wav(N_batches)