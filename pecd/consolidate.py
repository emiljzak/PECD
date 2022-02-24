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
        self.params     = params
        self.helicity   = params['helicity_consolidate']
        GridObjEuler    = wavefunction.GridEuler(   self.params['N_euler'],
                                                    self.params['N_batches'],
                                                    self.params['orient_grid_type'])
    
        self.grid_euler, self.N_Euler, self.N_per_batch  = GridObjEuler.read_euler_grid()


    def read_obs(self):
        barray = []
        bcoeffs_dict = {"L":[],"R":[]}

        for ibatch in range(0,self.params['N_batches']):
            bcoeffs_dict_ibatch = self.read_h5(ibatch)
            for sigma,b in bcoeffs_dict_ibatch.items():
                bcoeffs_dict[sigma].append(b)

            
            
        print(bcoeffs_dict)
        print("shape of bcoeffs_dict[L]:")
        print(np.shape(bcoeffs_dict['L']))


        for key,value in bcoeffs_dict.items():
            blist = value
            barray = np.concatenate(np.asarray(blist),axis=0)
            bcoeffs_dict[key] = barray


        return bcoeffs_dict


    def read_h5(self,ibatch):

        index_energy = self.params['index_energy'][0]
        index_time = self.params['index_time'][0]
        index_bcoeff = self.params['index_bcoeff']

        bcoeffs_dict = {}
        print(ibatch)
        with h5py.File(self.params['job_directory']+"bcoeffs_batch_"+str(ibatch)+".h5", 'r') as h5:
            G = h5.get('bcoefs_group')
            for sigma in self.helicity:

                bcoefs_arr = np.asarray(G.get("bcoefs"+sigma),dtype=float)
                bcoeffs_dict[sigma] = bcoefs_arr[:,index_time,index_energy,index_bcoeff]
                #print(list(G.items()))

        return bcoeffs_dict


    def calc_bcoeffs_av(self,bcoeff_dict):
        """Calculate orientation averaged b-coefficients for selected helicities"""
        bcoeff_av_dict = {}

        for sigma,barray in bcoeff_dict.items():
            Nomega = barray.shape[0]
            print("Nomega = " + str(Nomega))
            bav = np.zeros((barray.shape[1]),dtype=float)

            for n in range(barray.shape[1]):
                bav[n] = np.sum(barray[:,n])/Nomega
            
            bcoeff_av_dict[sigma] = bav
            print("Orientation-averaged b-coefficients for sigma = "+sigma)
            print(bav)
        return bcoeff_av_dict

    def calc_pecd(self,bcoeffs_dict):
        """Calculate multiphoton-PECD for a grid of orientations (beta,gamma) and a sequence of total photon numbers
        
            Use integrated Legendre expansion of increasing size for the calculation of appropriate coefficients.
        """
        
        coefficients = [1.0, -1.0/4.0, 1.0/8.0, -5.0/64.0, 7.0/128.0]

        Nphotons = self.params['Nmax_photons']
        
        for sigma,barray in bcoeffs_dict.items():
            Nomega = barray.shape[0]


        pecd = np.zeros((Nomega,Nphotons),dtype=float)

        for sigma,barray in bcoeffs_dict.items():

            if sigma == "L":
                sign = -1
            elif sigma == "R":
                sign = +1
            print("helicity = " + sigma)
            for nph in range(1,Nphotons+1):
                print("Number of photons = " + str(nph))
                for n in range(2*nph+1):
                    print("n = "+str(n))
                    pecd[:,nph-1] += sign*coefficients[n]*barray[:,n]

        print("PECD across orientations")
        print(pecd)
        return pecd


    def calc_pecd_av(self,pecd,b_av_dict):
        """Calculate multi-photon PECD averaged over orientations, for a sequence of total photon numbers"""

        Nphotons = self.params['Nmax_photons']

        pecd_av = np.zeros(Nphotons,dtype=float)

        for nph in range(1,Nphotons+1):
            pecd_av[nph-1] = np.sum(pecd[:,nph-1])
        
        print("PECD_av")
        print(pecd_av)
        return pecd_av


    def plot_bcoeffs_2D(self,barray,b_av):

        for n in range(barray.shape[1]):
            fig = plt.figure()
            grid_fig = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
            ax1 = fig.add_subplot(grid_fig[0, 0], projection='rectilinear')
            line_b = ax1.tricontourf( self.grid_euler[:,1],self.grid_euler[:,2],barray[:,n]/b_av[0],100,cmap = 'jet')
            plt.colorbar(line_b, ax=ax1, aspect=30) 
            fig.savefig(  "bcoeffs_map"+str(n)+".pdf"   )
            plt.close()


    def average_alpha(self):
        """Average the input b-coefficients over the alpha angle"""


    def check_b_symmetry(self,b_av_dict):
        """Check the symmetry of b-coefficients"""
        return 0

"""
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
    bcoeffs_dict = Obs.read_obs()
 
    #calculate orientation-averaged b-coefficients for given time, energy and helicities.
    b_av_dict   = Obs.calc_bcoeffs_av(bcoeffs_dict)
    pecd        = Obs.calc_pecd(bcoeffs_dict)
    pecd_av     = Obs.calc_pecd_av(pecd,b_av_dict)
    #Obs.plot_bcoeffs_2D(barray,b_av)
