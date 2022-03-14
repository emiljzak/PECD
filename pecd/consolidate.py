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

        #read 3D Euler grid
        GridObjEuler    = wavefunction.GridEuler(   self.params['N_euler'],
                                                    self.params['N_batches'],
                                                    self.params['orient_grid_type'])
    
        self.grid_euler, self.N_Euler, self.N_per_batch  = GridObjEuler.read_euler_grid()
        #generate 2D euler grid for alpha-averaged variables
        self.grid_euler2D, self.n_grid_euler_2d = GridObjEuler.gen_euler_grid_2D()   
    
    def read_Flm(self):
        """Reads Flm arrays for a sequeces of batches
        
        """
        
        Flm_dict = {"L":[],"R":[]}

        for ibatch in range(0,self.params['N_batches']):
            Flm_dict_ibatch = self.read_Flm_h5(ibatch)

            for sigma,f in Flm_dict_ibatch.items():
                Flm_dict[sigma].append(f)

        print(Flm_dict)
        print("shape of Flm_dict[L]:")
        print(np.shape(Flm_dict['L']))

        #merge Flm arrays from separate batches
        for key,value in Flm_dict.items():

            Flm_dict[key] = np.concatenate(np.asarray(value),axis=0)

        print("shape of Flm_dict[L] after concatenate:")
        print(np.shape(Flm_dict['L']))

        return Flm_dict


    def read_bcoeffs(self):
        barray = []
        bcoeffs_dict = {"L":[],"R":[]}

        for ibatch in range(0,self.params['N_batches']):
            bcoeffs_dict_ibatch = self.read_bcoeffs_h5(ibatch)
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


    def read_bcoeffs_h5(self,ibatch):

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

    def find_nearest_ind(self,array, value):
            array   = np.asarray(array)
            idx     = (np.abs(array - value)).argmin()
            return array[idx], idx

    def read_Flm_h5(self,ibatch):

        index_energy = self.params['index_energy'][0]
        index_time = self.params['index_time'][0]
        index_bcoeff = self.params['index_bcoeff']

        energy = params['legendre_params']['energy_grid'][index_energy]
        with open( self.params['job_directory'] +  "kgrid.dat" , 'r') as kgridfile:   
            kgrid = np.loadtxt(kgridfile)

        print("loaded kgrid: ")
        print(kgrid)
    
        #value of momentum at which we want to evaluate Flm
        val = np.sqrt(2.0*energy)
        #index of momentum at which we want ot evaluate Flm
        _,index_k = self.find_nearest_ind(kgrid,val)
        #print(index_k)
        #exit()
        Flm_dict = {}
        print("Reading ibatch = " + str(ibatch))
        with h5py.File(self.params['job_directory']+"Flm_batch_"+str(ibatch)+".h5", 'r') as h5:
            G = h5.get('Flm_group')
            for sigma in self.helicity:

                Flm_arr = np.asarray(G.get("Flm"+sigma),dtype=complex)

                Flm_dict[sigma] = Flm_arr[:,index_time,:,:,index_k] #F[omega,t,l,m,k]
                #print(list(G.items()))

        return Flm_dict


    def calc_Flm_alpha_av(self,Flm_dict):
        """Calculating alpha-averaged Flm for a selected set of energies and times
        We choose one energy and one time. The output is kept in sigma-resolved dictionary.
        """

        Flm_alpha_av_dict = {'L':[],
                            'R':[]}

        grid3D = self.grid_euler
        Ngrid3D = grid3D.shape[0]
        print("Ngrid3D = " + str(Ngrid3D))

        print(grid3D)

        #generate 2D euler grid for alpha-averaged variables
        grid2D = self.grid_euler2D
        Ngrid2D = grid2D.shape[0]   
        print("Ngrid2D = " + str(Ngrid2D))

        print(grid2D)


        for sigma,Flm in Flm_dict.items():

            if sigma == "L":
                sign = -1
            elif sigma == "R":
                sign = +1

            
            Flm_alpha_av = np.zeros((grid2D.shape[0],Flm.shape[1],Flm.shape[2]),dtype=complex)

            for betagamma in range(Ngrid2D):

                for ialpha in range(self.params['N_euler']+2):
                    Flm_alpha_av[betagamma,:,:] += Flm[betagamma+ialpha*Ngrid2D,:,:]

            Flm_alpha_av_dict[sigma] = Flm_alpha_av

        return Flm_alpha_av_dict,grid2D


    def calc_bcoeffs_Flm_alpha_av(self, Flm_alpha_av_dict):
        """Calculate b-coefficients from alpha-averaged Flm"""
        
        bcoeffs_arr     = np.zeros((Flm_alpha_av_dict["L"].shape[0],Flm_alpha_av_dict["L"].shape[1]),dtype = float)
        bcoeffs_dict    = { 'L':[],
                            'R':[]}

        for sigma,Flm in Flm_alpha_av_dict.items():
            Nomega = Flm.shape[0]
            print("Nomega = " + str(Nomega))
            for l in range(Flm.shape[1]):
                bcoeffs_arr[:,l] = Flm[:,l,0] * np.sqrt((2.0*float(l)+1)/(4.0*np.pi))

            bcoeffs_dict[sigma] = bcoeffs_arr
        return bcoeffs_dict

    def calc_bcoeffs_av(self,bcoeff_dict):
        """Calculate 3D orientation-averaged b-coefficients for selected helicities"""
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


    def plot_bcoeffs_Flm_2D(self,b_flm_alpha_av,grid2D):
        
        for sigma,barray in b_flm_alpha_av.items():

            print(grid2D)
            print(barray)
            print(barray.shape)
            #exit()
            for n in range(barray.shape[1]):
                fig = plt.figure()
                grid_fig = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
                ax1 = fig.add_subplot(grid_fig[0, 0], projection='rectilinear')
                line_b = ax1.tricontourf( grid2D[:,1],grid2D[:,2],barray[:,n],100,cmap = 'jet')
                plt.colorbar(line_b, ax=ax1, aspect=30) 
                fig.savefig(  "bcoeffs_flm_map"+str(n)+"_"+str(sigma)+".pdf"   )
                plt.close()
    

    def plot_bcoeffs_2D(self,barray,b_av):

        for n in range(barray.shape[1]):
            fig = plt.figure()
            grid_fig = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
            ax1 = fig.add_subplot(grid_fig[0, 0], projection='rectilinear')
            line_b = ax1.tricontourf( self.grid_euler[:,1],self.grid_euler[:,2],barray[:,n]/b_av[0],100,cmap = 'jet')
            plt.colorbar(line_b, ax=ax1, aspect=30) 
            fig.savefig(  "bcoeffs_map"+str(n)+".pdf"   )
            plt.close()


    def plot_pecd_2D(self,barray,b_av):
        """Generate 2D plot of PECD for a sequence of photon numbers"""
        for n in range(barray.shape[1]):
            fig = plt.figure()
            grid_fig = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
            ax1 = fig.add_subplot(grid_fig[0, 0], projection='rectilinear')
            line_b = ax1.tricontourf( self.grid_euler[:,1],self.grid_euler[:,2],barray[:,n]/b_av[0],100,cmap = 'jet')
            plt.colorbar(line_b, ax=ax1, aspect=30) 
            fig.savefig(  "pecd_map"+str(nph)+".pdf"   )
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
    bcoeffs_dict = Obs.read_bcoeffs()
 
    #read Flm's and do alpha-averaging
    Flm_dict = Obs.read_Flm()
    Flm_alpha_av_dict,grid2D = Obs.calc_Flm_alpha_av(Flm_dict)
    
    #calculate b-coeffs from alpha-averaged Flm
    b_flm_alpha_av = Obs.calc_bcoeffs_Flm_alpha_av( Flm_alpha_av_dict)
    Obs.plot_bcoeffs_Flm_2D(b_flm_alpha_av,grid2D)


    #calculate alpha-averaged b-coeffs the legendre expansion 
    #b_alpha_av = Obs.calc_bcoeffs_alpha_av(bcoeffs_dict)
    #calculate 3D orientation-averaged b-coeffs 
    #b_av_dict   = Obs.calc_bcoeffs_av(bcoeffs_dict)
    #Obs.plot_bcoeffs_2D(b_alpha_av,b_av_dict)
    exit()

    #calculate orientation-averaged b-coefficients for given time, energy and helicities.
    b_av_dict   = Obs.calc_bcoeffs_av(bcoeffs_dict)
    pecd        = Obs.calc_pecd(bcoeffs_dict)
    pecd_av     = Obs.calc_pecd_av(pecd,b_av_dict)
    #
