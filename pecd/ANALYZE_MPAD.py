import numpy as np

import quaternionic
import spherical
import PLOTS 

from sympy.physics.quantum.spin import Rotation

from sympy import pi, symbols
from sympy import N


def test_wigner():
    """ Test wigner functions: orthogonality, symmetry and plots (2D slices of known functions). Decide how to store them """

    Jmax = 4
    wigner = spherical.Wigner(Jmax)

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




def analyze_Wav(N_batches):

    working_dir = "/gpfs/cfel/cmi/scratch/user/zakemil/PECD/tests/molecules/d2s/"

    with open( working_dir  + "grid_W_av" , 'r') as gridfile:   
        grid = np.loadtxt(gridfile)

    grid = grid.T
    Wav = np.zeros((grid.shape[0],grid.shape[0]), dtype = float)  
    Wavi = np.zeros((grid.shape[0],grid.shape[0]), dtype = float)

    for ibatch in range(N_batches):
        with open( working_dir + "W_av_3D_" + str(ibatch), 'r') as Wavfile:   
            Wavi = np.loadtxt(Wavfile)
        Wav += Wavi

    with open( working_dir  + "W_av_3D" , 'w') as Wavfile:   
        np.savetxt(Wavfile, Wav, fmt = '%10.4e')

    grid = grid.T
    PLOTS.plot_2D_polar_map(Wav,grid[1],grid[0],100)



N_batches = 3

test_wigner()

#analyze_Wav(N_batches)