import os
import sys
import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

import h5py
import spherical
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

        lmax = self.params['bound_lmax']

        for sigma,Flm in Flm_alpha_av_dict.items():
            Nomega = Flm.shape[0]
            barray = np.zeros((Nomega,2*self.params['Nmax_photons']),dtype = complex)
            print("Nomega = " + str(Nomega))
            for L in range(2*self.params['Nmax_photons']):

                for l1 in range(lmax):
                    for l2 in range(lmax):
                        if l1+l2 >= L and np.abs(l1-l2) <= L:
                            tau = 0.0+1j * 0.0
                            for m1 in range(-l1,l1+1):
                                for m2 in range(-l2,l2+1):
                                    tau +=  (-1)**m1 * np.conjugate(Flm[:,l1,l1+m1]) * Flm[:,l2,l2+m2] *  spherical.clebsch_gordan(l1,0,l2,0,L,0) * spherical.clebsch_gordan(l1,-1*m1,l2,m2,L,0) 

                            barray[:,L] += tau * (-1.0)**(l2) * (1j)**(l1+l2) * np.sqrt((2.0*float(l1)+1)*(2.0*float(l2)+1))

           # for l in range(Flm.shape[1]):
           #     bcoeffs_arr[:,l] = barray[:,l] * np.sqrt((2.0*float(l)+1)/(4.0*np.pi))
            print(barray)
            bcoeffs_dict[sigma] = 8.0 * np.pi**2 * barray
        return bcoeffs_dict


    def calc_bcoeffs_alpha_av(self,bcoeffs_dict):
        """Calculating alpha-averaged b for a selected set of energies and times
        We choose one energy and one time. The output is kept in sigma-resolved dictionary.
        """

        bcoeffs_alpha_av_dict = {'L':[],
                            'R':[]}

        grid3D = self.grid_euler
        Ngrid3D = grid3D.shape[0]
        print("Ngrid3D = " + str(Ngrid3D))


        #generate 2D euler grid for alpha-averaged variables
        grid2D = self.grid_euler2D
        Ngrid2D = grid2D.shape[0]   
        print("Ngrid2D = " + str(Ngrid2D))

        for sigma,barray in bcoeffs_dict.items():

            if sigma == "L":
                sign = -1
            elif sigma == "R":
                sign = +1

            bcoeffs_alpha_av = np.zeros((grid2D.shape[0],barray.shape[1]),dtype=complex)

            for betagamma in range(Ngrid2D):

                for ialpha in range(self.params['N_euler']+2):
                    bcoeffs_alpha_av[betagamma,:] += barray[betagamma+ialpha*Ngrid2D,:]

            bcoeffs_alpha_av_dict[sigma] = bcoeffs_alpha_av/(self.params['N_euler']+2)

        return bcoeffs_alpha_av_dict,grid2D


    def calc_bcoeffs_av(self,bcoeff_dict,grid3D):
        """Calculate 3D orientation-averaged b-coefficients for selected helicities"""
        
        #calculate sin(beta) on the euler grid
        sinbeta = np.sin(grid3D[:,1])
        
        bcoeff_av_dict = {}

        for sigma,barray in bcoeff_dict.items():
            Nomega = barray.shape[0]
            print("Nomega = " + str(Nomega))
            bav = np.zeros((barray.shape[1]),dtype=float)

            for n in range(barray.shape[1]):
                bav[n] = np.sum(sinbeta[:]*barray[:,n])/Nomega
            
            bcoeff_av_dict[sigma] = bav
            print("Orientation-averaged b-coefficients for sigma = "+sigma)
            print(bav/bav[0])
            #save b-coeffs
            
        print("2b_1/b_0:")
        print(200*bcoeff_av_dict["R"][1]/bcoeff_av_dict["R"][0])
        with open(  self.params['job_directory'] +  "bcoeffs_averaged.dat" , 'w') as bcoeffile:
            for sigma,val in bcoeff_av_dict.items(): 
 
                bcoeffile.write(str(sigma) +  " ".join('{:12.8f}'.format(val[n]) for n in range(val.shape[0]))+"\n") 

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
                for n in range(1,2*nph+1,2):
                    print("n = "+str(n))
                    pecd[:,nph-1] += sign*coefficients[n]*barray[:,n].real

        #print("PECD across orientations")
        #print(pecd)
        return pecd


    def calc_pecd_av(self,pecd,b_av_dict):
        """Calculate multi-photon PECD averaged over orientations, for a sequence of total photon numbers"""
        bav = b_av_dict["R"][0]
        Nphotons = self.params['Nmax_photons']

        pecd_av = np.zeros(Nphotons,dtype=float)

        for nph in range(1,Nphotons+1):
            pecd_av[nph-1] = np.sum(pecd[:,nph-1])
        
        print("PECD_av")
        print(pecd_av)
        return pecd_av


    def plot_bcoeffs_Flm_2D(self,b_flm_alpha_av_dict,grid2D):
        
        for sigma,barray in b_flm_alpha_av_dict.items():

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
    

    def plot_bcoeffs_2D(self,b_alpha_av_dict,b_av_dict,grid2D,name):
        
        #<b_0>
        bav = b_av_dict["R"][0]


        for sigma,barray in b_alpha_av_dict.items():

            print(grid2D)
            print(barray)
            print(barray.shape)
            print("bav = " + str(bav))
            #exit()
            for n in range(barray.shape[1]):
                fig = plt.figure()
                grid_fig = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
                ax1 = fig.add_subplot(grid_fig[0, 0], projection='rectilinear')
                line_b = ax1.tricontourf( grid2D[:,2],grid2D[:,1],100*(barray[:,n].real)/bav,100,cmap = 'jet')
                plt.colorbar(line_b, ax=ax1, aspect=30) 
                fig.savefig(  "bcoeffs_"+str(name)+"_"+str(n)+"_"+str(sigma)+".pdf"   )
                plt.close()
    

    def plot_pecd_2D(self,pecd_alpha_av,b_av_dict,grid2D):
        """Generate 2D plot of PECD for a sequence of photon numbers"""
        bav = b_av_dict["R"][0]
        for n in range(pecd_alpha_av.shape[1]):
            fig = plt.figure()
            grid_fig = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
            ax1 = fig.add_subplot(grid_fig[0, 0], projection='rectilinear')
            line_b = ax1.tricontourf( grid2D[:,2],grid2D[:,1],100*pecd_alpha_av[:,n]/bav,100,cmap = 'jet')
            plt.colorbar(line_b, ax=ax1, aspect=30) 
            fig.savefig(  "pecd_map_nph"+str(n+1)+".pdf"   )
            plt.close()

    def average_alpha(self):
        """Average the input b-coefficients over the alpha angle"""


    def check_b_symmetry(self,b_av_dict):
        """Check the symmetry of b-coefficients"""
        return 0

    def check_sizes(self):
        """This module produces a list of unfinished jobs based on wavepacket file size"""
        missing_filesL = []
        awkward_filesL = []
        missing_filesR = []
        awkward_filesR = []

        default_size_min = 4*10**8
        default_size_max = 4.2*10**8
        # The directory that we are interested in
        myPath = params['job_directory']
        print(self.N_Euler)
        for i in range(self.N_Euler):
            if os.path.isfile(myPath +"wavepacketL_"+str(i)+".h5"):
                fsize =  os.path.getsize(myPath+"wavepacketL_"+str(i)+".h5")
                print(fsize)
                if fsize < default_size_min or fsize > default_size_max:
                    awkward_filesL.append(i)

            else:
                missing_filesL.append(i)
            if os.path.isfile(myPath +"wavepacketR_"+str(i)+".h5"):
                fsize =  os.path.getsize(myPath+"wavepacketR_"+str(i)+".h5")
                print(fsize)
                if fsize < default_size_min or fsize > default_size_max:
                    awkward_filesR.append(i)
            else:
                missing_filesR.append(i)


        print("L: awkward files:")
        for ielem,elem in enumerate(awkward_filesL):
            if ielem%10 == 0 and ielem>0:
                print(awkward_filesL[ielem-10:ielem])
            if ielem > len(awkward_filesL-10):
                print(awkward_filesL[ielem:])

        print("R: awkward files:")
        for ielem,elem in enumerate(awkward_filesR):
            if ielem%10 == 0 and ielem>0:
                print(awkward_filesR[ielem-10:ielem])                      
            if ielem > len(awkward_filesR-10):
                print(awkward_filesR[ielem:])


        print("L: missing files:")
        for ielem,elem in enumerate(missing_filesL):
            if ielem%10 == 0 and ielem>0:
                print(missing_filesL[ielem-10:ielem])
            if ielem > len(missing_filesL-10):
                print(missing_filesL[ielem:])

        print("R: missing files:")
        for ielem,elem in enumerate(missing_filesR):
            if ielem%10 == 0 and ielem>0:
                print(missing_filesR[ielem-10:ielem])
            if ielem > len(missing_filesR-10):
                print(missing_filesR[ielem:])

        #print("R - missing and awkward files lists, respectively:")
        #print(missing_filesR)
        #print(awkward_filesR)
        #print("L - missing and awkward files lists, respectively:")
        #print(missing_filesL)
        #print(awkward_filesL)
        exit()
        return missing_files,awkward_files
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
    os.chdir(sys.argv[2])
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

    if params['check_sizes'] == True:
        print("commencing with size check of wavepackets...")
        Obs.check_sizes()

    """ ===== b-coeffs using Flm amplitudes and an anlytic formula ===="""
    #read Flm's and do alpha-averaging
    #Flm_dict = Obs.read_Flm()
    #Flm_alpha_av_dict,grid2D = Obs.calc_Flm_alpha_av(Flm_dict)
    
    #calculate b-coeffs from alpha-averaged Flm
    #bcoeffs_flm_alpha_av_dict = Obs.calc_bcoeffs_Flm_alpha_av( Flm_alpha_av_dict)
    #b_av_dict   = Obs.calc_bcoeffs_av(bcoeffs_flm_alpha_av_dict)
    #Obs.plot_bcoeffs_2D(bcoeffs_flm_alpha_av_dict,b_av_dict,grid2D,"flm")

    """ ===== b-coeffs using numerical Legendre expansion ===="""
    bcoeffs_dict = Obs.read_bcoeffs()
    #calculate alpha-averaged b-coeffs the legendre expansion 
    bcoeffs_alpha_av_dict,grid2D = Obs.calc_bcoeffs_alpha_av(bcoeffs_dict)
    #calculate 3D orientation-averaged b-coeffs 
    b_av_dict   = Obs.calc_bcoeffs_av(bcoeffs_dict,Obs.grid_euler)
    
  
    Obs.plot_bcoeffs_2D(bcoeffs_alpha_av_dict,b_av_dict,grid2D,"leg")
    

    #calculate multi-photon alpha-averaged PECD for (beta,gamma) orientations


    #calculate orientation-averaged b-coefficients for given time, energy and helicities.
    pecd_alpha_av        = Obs.calc_pecd(bcoeffs_alpha_av_dict)

    Obs.plot_pecd_2D(pecd_alpha_av,b_av_dict,grid2D )
    pecd_av     = Obs.calc_pecd_av(pecd_alpha_av,b_av_dict)
    #
    print("b1(R)= " + str(b_av_dict["R"][1]))
    print("b1(L)= " + str(b_av_dict["L"][1]))