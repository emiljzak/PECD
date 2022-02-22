import os
import sys
import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

import h5py

import wavefunction

class Avobs:
    def __init__(self,params):
        self.params = params
        self.helicity = params['helicity']
        GridObjEuler = wavefunction.GridEuler(  self.params['N_euler'],
                                                self.params['N_batches'],
                                                self.params['orient_grid_type'])
    
        self.grid_euler, self.N_Euler, self.N_per_batch  = GridObjEuler.read_euler_grid()


    def read_obs(self):
        barray = []
        for ibatch in range(0,self.params['N_batches']):
            bb = self.read_h5(ibatch)
            #print(bb)
            barray.append(bb)
        
        barray = np.asarray(barray,dtype=float)
        #print("shape of barray " + str(barray.shape))

        #print(barray.ravel())
        return barray.ravel()


    def read_h5(self,ibatch):


        index_energy = self.params['index_energy'][0]
        index_time = self.params['index_time'][0]
        index_bcoeff = self.params['index_bcoeff'][0]

        with h5py.File(self.params['job_directory']+"bcoeffs_batch_"+str(ibatch)+".h5", 'r') as h5:
            G = h5.get('bcoefs_group')
            bcoefs_arr = np.array(G.get('bcoefs'))
            print(bcoefs_arr.shape)
            barray = bcoefs_arr[:,index_time,index_energy,index_bcoeff]
            #print(list(G.items()))
        
        return barray


    def calc_bcoeffs(self):
        pass


    def plot_bcoeffs_2D(self,barray):

        fig = plt.figure(figsize=(100,100), dpi=200,
                        constrained_layout=True)
        grid_fig = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

        ax1 = fig.add_subplot(grid_fig[0, 0], projection='rectilinear')

        print(self.grid_euler.shape)
        #exit()

        print(self.grid_euler[:,1])
        print(self.grid_euler[:,2])
        print(barray)
        #exit()
        #ax1.plot(self.grid_euler[:,1],barray)
        plot_cont_1 = ax1.tricontourf( self.grid_euler[:,1],self.grid_euler[:,2],barray,100,cmap = 'jet')
       

        #plt.show()
        #plt.close()
        fig.savefig(    fname       = "bcoeffs_map.pdf",
                        dpi         = 200       )



    """


    Wav = np.zeros((grid_theta.shape[0],grid_r.shape[0]), dtype = float)




    if params['density_averaging'] == True:
        WDMATS  = gen_wigner_dmats( N_Euler, 
                                    params['Jmax'], 
                                    grid_euler)

        #calculate rotational density at grid (alpha, beta, gamma) = (n_grid_euler, 3)
        grid_rho, rho = ROTDENS.calc_rotdens(   grid_euler,
                                                WDMATS,
                                                params) 


          
        print(n_grid_euler_2d)
        grid_rho, rho = ROTDENS.calc_rotdens( grid_euler_2d,
                                    WDMATS,
                                    params) 
        print("shape of rho")
        print(np.shape(rho))
        #print(rho.shape)
        #PLOTS.plot_rotdens(rho[:].real, grid_euler_2d)
    """

if __name__ == "__main__":   

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    print(" ")
    print("---------------------- START CONSOLIDATE --------------------")
    print(" ")
    print(sys.argv)
    os.chdir(sys.argv[1])
    path = os.getcwd()
    print(path)
    # read input_propagate
    with open('input_prop', 'r') as input_file:
        params_prop = json.load(input_file)
    #read input_analyze
    with open('input_analyze', 'r') as input_file:
        params_analyze = json.load(input_file)
    with open('input_consolidate', 'r') as input_file:
        params_consolidate = json.load(input_file)

    #combine inputs
    params = {}
    params.update(params_prop)
    params.update(params_analyze)
    params.update(params_consolidate)

    Obs = Avobs(params)
    barray = Obs.read_obs()

    Obs.plot_bcoeffs_2D(barray)
