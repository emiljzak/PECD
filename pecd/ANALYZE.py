import numpy as np
import json
from scipy.special import sph_harm

import time
import os
import sys


import MAPPING
import GRID
import CONSTANTS
import PLOTS

import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt


def spharm(l,m,theta,phi):
    return sph_harm(m, l, phi, theta)
    

def read_wavepacket(filename, itime, Nbas):

    coeffs = []
    #print(itime)
    
    with open(filename, 'r', ) as f:
        for _ in range(itime):
            next(f)
        for line in f:
            words   = line.split()
            for ivec in range(2*Nbas):
                coeffs.append(float(words[1+ivec]))

        """
        #print(float(record[0][1]))
        for line in itertools.islice(f, itime-1, None):
            print(np.shape(line))
            print(type(line))
            #print(line)
        """
    """
    for line in fl:

        i       = int(words[0])
        n       = int(words[1])
        xi      = int(words[2])
        l       = int(words[3])
        m       = int(words[4])
        c       = []
        for ivec in range(nvecs):
            c.append(float(words[5+ivec]))
        coeffs.append([i,n,xi,l,m,np.asarray(c)])
    """
    return coeffs

def pull_helicity(params):
    if params['field_type']['function_name'] == "fieldRCPL":
        helicity = "R"
    elif params['field_type']['function_name'] == "fieldLCPL":
        helicity = "L"  
    elif params['field_type']['function_name'] == "fieldLP":
        helicity = "0"
    else:
        raise ValueError("Incorect field name")
    return helicity

class analysis:
    def __init__(self,wavepacket,params,maparray):
        self.wavepacket = wavepacket
        self.params = params
        self.maparray = maparray

    def calc_tgrid(self):
        print("Setting up time-grid")

        tgrid = np.linspace(    self.params['t0'] * time_to_au, 
                                self.params['tmax'] * time_to_au, 
                                int((self.params['tmax']-self.params['t0'])/self.params['dt']+1), 
                                endpoint = True )
        dt = self.params['dt'] * time_to_au
        return tgrid, dt


    def calc_plot_times(self,tgrid,dt,plot_times):

        plot_index = []
        for index,item in enumerate(plot_times):
            if int( item * time_to_au / dt ) > len(tgrid):
                print("removing time: " + str(item) + " from plotting list. Time exceeds the propagation time-grid!")
            else:
                plot_index.append( int(item * time_to_au / dt) )
        print("Final list of plottime indices in tgrid:")
        print(plot_index)

        return plot_index


    def calc_rho2D(self, psi, polargrid, funcpar, Nbas_chi, lmax):

        plane       = funcpar['plane']
        coeff_thr   = funcpar['coeff_thr']

    
        rho = np.zeros((polargrid[0].shape[0],polargrid[1].shape[1]), dtype=complex)

        for elem in plane:
            if elem == "XY":
                print("Evaluation plane for rho2D: " + elem)
                theta0 = np.pi / 2.0
            elif elem == "XZ":
                print("Evaluation plane for rho2D: " + elem)
                phi0 =  0.0

                #calculate spherical harmonics on the angular grid for all quantum numbers
                Ymat = np.zeros((lmax+1,2*lmax+1,polargrid[0].shape[0],polargrid[1].shape[1]),dtype=complex)
                
                for l in range(lmax+1):
                    for m in range(-l,l+1):
                        Ymat[l,l+m,:,:] =  spharm(l, m, polargrid[1], phi0) 
                #print(Ymat)
                #exit()
                icoeff = 0
                
                for xi in range(Nbas_chi):
                    chi_rgrid = chilist[xi](polargrid[0].ravel())
                    chi_rgrid = chi_rgrid.reshape(polargrid[0].shape[0],polargrid[1].shape[1])
                    #print(chi_rgrid)
                    for l in range(lmax+1):
                        for m in range(-l,l+1):
                            #if np.abs(psi[ielem]) > coeff_thr:
                            #print(psi[icoeff])
                            rho += psi[icoeff] * Ymat[l,l+m,:,:]
                            icoeff += 1
                    rho += chi_rgrid * rho

                #print("icoeff = " + str(icoeff))
                #exit()

     
            elif elem == "YZ":
                print("Evaluation plane for rho2D: " + elem)
                phi0 =  np.pi/2
             

            elif len(elem) == 3:
                print("Evaluation plane defined by normal vector: " + str(elem))
                #to implement
                exit()
            else:
                raise ValueError("incorrect evaluation plane for rho2D")

        

        return np.abs(rho)**2/np.max(np.abs(rho)**2)

    def rho2D(self,funcpars):
        print("Calculating 2D electron density")
        
        """ set up 1D grids """

        if funcpars['r_grid']['type'] == "manual":
            rtuple  = (funcpars['r_grid']['rmin'], funcpars['r_grid']['rmax'], funcpars['r_grid']['npts'])
        elif funcpars['r_grid']['type'] == "automatic":
            rmax = 0.0
            for elem in self.params['FEMLIST']:
                rmax += elem[0] * elem[2]
            rtuple = (0.0, rmax, funcpars['r_grid']['npts'] )
        else:
            raise ValueError("incorrect radial grid type")
        
        thtuple     = funcpars['th_grid']

        rgrid1D         = np.linspace(rtuple[0], rtuple[1], rtuple[2], endpoint=True, dtype=float)
        unity_vec       = np.linspace(0.0, 1.0, thtuple[2], endpoint=True, dtype=float)
        thgrid1D        = thtuple[1] * unity_vec


        """ generate 2D meshgrid for storing rho2D """
        polargrid = np.meshgrid(rgrid1D, thgrid1D, indexing='ij')

        """ set up time grids for evaluating rho2D """
        tgrid,dt = self.calc_tgrid()

        plot_index = self.calc_plot_times(tgrid,dt,funcpars['plot_times']) #plot time indices

        tgrid_plot = tgrid[plot_index]


        for itime, t in zip(plot_index,list(tgrid_plot)):

            print(  "Generating plot at time = " + str('{:6.2f}'.format(t/time_to_au) ) +\
                    " " + str( self.params['time_units']) + " ----- " +\
                    "time index = " + str(itime) )

            #psi = self.wavepacket[itime,:]

            
            rho = self.calc_rho2D(  psi, 
                                    polargrid, 
                                    funcpars,  
                                    self.params['Nbas_chi'], 
                                    self.params['bound_lmax'])
            #print(rho)
            #exit()
            if funcpars['plot'] == True:

                fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
                spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
                axradang_r = fig.add_subplot(spec[0, 0], projection='polar')
                line_angrad_r = axradang_r.contourf(polargrid[1],polargrid[0], rho, 
                                                        100, cmap = 'jet', vmin=0.0, vmax=1.0) #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
                    #plt.colorbar(line_angrad_r, ax=axradang_r, aspect=30)
                axradang_r.set_rlabel_position(100)
                    #axradang_r.set_yticklabels(list(str(np.linspace(rmin,rmax,5.0)))) # set radial tick label
                    #plt.legend()   
                plt.show()  
                #PLOTS.plot_2D_polar_map(rho,polargrid[1],polargrid[0],100,self.params)


                #PLOTS.plot_rho2D(   self.params,  )

            if funcpars['save'] == True:

                with open( params['job_directory'] +  "rho2D" + "_"+ helicity + ".dat" , 'w') as rhofile:   
                    np.savetxt(rhofile, rho2D, fmt = '%10.4e')

            



if __name__ == "__main__":   

    start_time_total = time.time()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    print(" ")
    print("---------------------- START ANALYZE --------------------")
    print(" ")
    

    ibatch = int(sys.argv[1]) # id of batch of Euler angles grid run
    os.chdir(sys.argv[2])
    path = os.getcwd()

    print("dir: " + path)

    # read input_propagate

    with open('input_prop', 'r') as input_file:
        params_prop = json.load(input_file)


    #read input_analyze
    with open('input_analyze', 'r') as input_file:
        params_analyze = json.load(input_file)

    #combine inputs
    params = {}
    params.update(params_prop)
    params.update(params_analyze)

    print(" ")
    print("---------------------- INPUT ECHO --------------------")
    print(" ")

    for key, value in params.items():
        print(key, ":", value)



    time_to_au = CONSTANTS.time_to_au[ params['time_units'] ]
    itime = int(params['analyze_time'] / params['dt'])

    maparray_global, Nbas_global = MAPPING.GENMAP_FEMLIST(  params['FEMLIST'],
                                                            params['bound_lmax'],
                                                            params['map_type'],
                                                            params['job_directory'] )

    Gr, Nr                       = GRID.r_grid(             params['bound_nlobs'], 
                                                            params['bound_nbins'] , 
                                                            params['bound_binw'],  
                                                            params['bound_rshift'] )

    """ Read grid of Euler angles"""
    with open( "grid_euler.dat" , 'r') as eulerfile:   
        grid_euler = np.loadtxt(eulerfile)

    grid_euler = grid_euler.reshape(-1,3)

    N_Euler = grid_euler.shape[0]

    N_per_batch = int(N_Euler/params['N_batches'])

    maparray_chi, params['Nbas_chi'] = MAPPING.GENMAP_FEMLIST( params['FEMLIST'],  0, \
                                params['map_type'], path )

    helicity = pull_helicity(params)


    Gr_prim, Nr_prim = GRID.r_grid_prim(    params['bound_nlobs'], 
                                            params['bound_nbins'], 
                                            params['bound_binw'], 
                                            params['bound_rshift'] )




    chilist = PLOTS.interpolate_chi(    Gr_prim, 
                                        params['bound_nlobs'], 
                                        params['bound_nbins'], 
                                        params['bound_binw'], 
                                        maparray_chi)

    """
    grid_theta, grid_r = calc_grid_for_FT(params)

    Wav = np.zeros((grid_theta.shape[0],grid_r.shape[0]), dtype = float)


    #which grid point corresponds to the radial cut-off?
    ipoint_cutoff = np.argmin(np.abs(Gr.ravel()- params['rcutoff']))
    print("ipoint_cutoff = " + str(ipoint_cutoff))


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


    print("=====================================")
    print("==post-processing of the wavepacket==")
    print("====================================="+"\n")

    for irun in range(ibatch * N_per_batch, (ibatch+1) * N_per_batch):
        print("processing grid point: " + str(irun) + " " + str(grid_euler[irun]) )

        alpha   = grid_euler[irun][0]
        beta    = grid_euler[irun][1]
        gamma   = grid_euler[irun][2]

        if params['density_averaging'] == True:
            print( "Rotational density at point " + str([alpha, beta, gamma]) + " is: " + str(rho[irun]))

        #read wavepacket from file
        file_wavepacket      =  params['job_directory'] + params['wavepacket_file'] + helicity + "_" + str(irun) + ".dat"

        psi                  = read_wavepacket(file_wavepacket, itime, Nbas_global)


        print("=========")
        print("==Plots==")
        print("=========")
        
        analysis_obj = analysis(psi,params,maparray_global)
        
        for elem in params['analyze_functions']:

            #call function by name given in the dictionary
            func = getattr(analysis_obj,elem['name'])
            print("Calling analysis function: " + str(elem['name']))
            func(elem)
        
        exit()






        if params['FT_method']    == "FFT_cart":
            calc_fftcart_psi_3d(params, maparray_global, Gr, psi, chilist)

        elif params['FT_method']  == "FFT_hankel":

            #calculate partial waves on radial grid
            Plm         = calc_partial_waves(chilist, grid_r, params['bound_lmax'], psi, maparray_global, maparray_chi, ipoint_cutoff)

            #calculate Hankel transforms on appropriate k-vector grid
            Flm, kgrid  = calc_hankel_transforms(Plm, grid_r)
            #gamma = 5.0*np.pi/8.0  #- to set fixed phi0 for FT
            params['analyze_mode']    = "2D-average" #3D, 2D-average
            params['nphi_pts']        = 50 #number of phi points for the integration over tha azimuthal angle.
            if params['analyze_mode'] == "2D-average":
                Wav, kgrid = calc_W_2D_av_phi_num(params, maparray_chi, maparray_global, psi, chilist)

            else:
                FT, kgrid   = calc_FT_3D_hankel(    Plm, Flm, kgrid, params['bound_lmax'], 
                                                grid_theta, grid_r, maparray_chi, 
                                                maparray_global, psi, chilist, gamma )
            
                if params['density_averaging'] == True:
                    #plot_W_3D_num(params, maparray_chi, maparray_global, psi, chilist, gamma)
                    Wav += float(rho[irun]) * np.abs(FT)**2
                    #PLOTS.plot_2D_polar_map(np.abs(FT)**2,grid_theta,kgrid,100)

                else:
                    print("proceeding with uniform rotational density")
                    Wav += np.abs(FT)**2

    with open( params['job_directory'] +  "W" + "_"+ helicity + "_av_3D_"+ str(ibatch) , 'w') as Wavfile:   
        np.savetxt(Wavfile, Wav, fmt = '%10.4e')

    with open( params['job_directory'] + "grid_W_av", 'w') as gridfile:   
        np.savetxt(gridfile, np.stack((kgrid.T,grid_theta.T)), fmt = '%10.4e')

    PLOTS.plot_pad_polar(params,params['k_list_pad'],helicity)