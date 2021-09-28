import numpy as np
import json

import time
import os
import sys


import MAPPING
import GRID
import CONSTANTS
import PLOTS


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
    def __init__(self,wavepacket,params):
        self.wavepacket = wavepacket
        self.params = params

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


    def rho2D(self,pars):
        print("Calculating 2D electron density")
        
        if pars['r_grid']['type'] == "manual":
            rgrid = (pars['r_grid']['rmin'], pars['r_grid']['rmax'], pars['r_grid']['npts'])
        elif pars['r_grid']['type'] == "automatic":
            rmax = 0.0
            for elem in self.params['FEMLIST']:
                rmax += elem[0] * elem[2]
            rgrid = (0.0, rmax, pars['r_grid']['npts'] )
        else:
            raise ValueError("incorrect radial grid type")
        
        th_grid = pars['th_grid']

        tgrid,dt = self.calc_tgrid()

        print(pars['plot_times'])

        plot_index = self.calc_plot_times(tgrid,dt,pars['plot_times']) #plot time indices

        tgrid_plot = tgrid[plot_index]


        for itime, t in zip(plot_index,list(tgrid_plot)):

            print(  "Generating plot at time = " + str('{:6.2f}'.format(t/time_to_au) ) +\
                    " " + str( self.params['time_units']) + " ----- " +\
                    "time index = " + str(itime) )

            #psi = self.wavepacket[itime,:]


            """rho = calc_rho2D(self,psi, rgrid,
                                    th_grid,
                                    psi, 
                                    maparray, 
                                    Gr_all, t, 
                                    chilist, 
                                    ieuler)

            if pars['plot'] == True:


                PLOTS.plot_2D_polar_map(rho,100,params)


                PLOTS.plot_rho2D(   self.params,  )

            if pars['save'] == True:

                with open( params['job_directory'] +  "rho2D" + "_"+ helicity + ".dat" , 'w') as rhofile:   
                    np.savetxt(rhofile, rho2D, fmt = '%10.4e')

            """



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

    maparray_chi, Nbas_chi = MAPPING.GENMAP_FEMLIST( params['FEMLIST'],  0, \
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
        
        analysis_obj = analysis(psi,params)
        
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