import numpy as np

import quaternionic
import spherical
import PLOTS 

from sympy.physics.quantum.spin import Rotation

from sympy import pi, symbols
from sympy import N

def test_wigner():
    l = 1
    wigner = spherical.Wigner(l)

    alpha = 0.4 * np.pi
    beta = 0.5 * np.pi 
    gamma = 0.2 * np.pi
    #alpha, beta, gamma = symbols('alpha beta gamma')
    #rot = Rotation.D(1, 1, 0, alpha, beta, gamma)
    #print(N(rot.evalf(subs={alpha:pi, beta:pi/2, gamma:0})))
    R = quaternionic.array.from_euler_angles(alpha, beta, gamma)
    #R=[1.0,1.0,0.0,1.0]
    D = wigner.D(R)
    print(wigner.Dindex(1,0,-1))
    print(D)
    #D[wigner.Dindex(ell, mp, m)]

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