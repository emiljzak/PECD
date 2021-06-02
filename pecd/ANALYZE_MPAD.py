import numpy as np
import PLOTS 



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

    grid = grid.T
    PLOTS.plot_2D_polar_map(Wav,grid[1],grid[0],100)



N_batches = 3

analyze_Wav(N_batches)