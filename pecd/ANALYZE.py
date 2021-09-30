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
import matplotlib
from matplotlib.ticker import FormatStrFormatter

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
    def __init__(self,params):
        self.params = params

    def calc_tgrid(self):
        print("Setting up time-grid")

        tgrid   = np.linspace(    self.params['t0'] * time_to_au, 
                                    self.params['tmax'] * time_to_au, 
                                    int((self.params['tmax']-self.params['t0'])/self.params['dt']+1), 
                                    endpoint = True )
        dt      = self.params['dt'] * time_to_au
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


    def split_word(self,word):
        return [char for char in word]
      
      

    def rho2D(self,funcpars):
        print("Calculating 2D electron density")
        
        irun = self.params['irun']
        helicity = self.params['helicity'] 

        """ set up 1D grids """

        if funcpars['r_grid']['type'] == "manual":
            rtuple  = (funcpars['r_grid']['rmin'], funcpars['r_grid']['rmax'], funcpars['r_grid']['npts'])
        elif funcpars['r_grid']['type'] == "automatic":
            rmax = 0.0
            for elem in self.params['FEMLIST']:
                rmax += elem[0] * elem[2]
            rtuple = (0.0, rmax, 2*int(rmax) )
        else:
            raise ValueError("incorrect radial grid type")
        
        thtuple     = funcpars['th_grid']

        rgrid1D         = np.linspace(rtuple[0], rtuple[1], rtuple[2], endpoint=True, dtype=float)
        unity_vec       = np.linspace(0.0, 1.0, thtuple[2], endpoint=True, dtype=float)
        thgrid1D        = thtuple[1] * unity_vec

        funcpars['rtulpe'] = rtuple
        funcpars['thtuple'] = thtuple
        
        #for xi in range(self.params['Nbas_chi']):

        #    plt.plot(rgrid1D,chilist[xi](rgrid1D))

        #plt.show()
        #exit()
        
        """ generate 2D meshgrid for storing rho2D """
        polargrid = np.meshgrid(rgrid1D, thgrid1D)

        """ set up time grids for evaluating rho2D """
        tgrid,dt = self.calc_tgrid()

        plot_index = self.calc_plot_times(tgrid,dt,funcpars['plot_times']) #plot time indices

        tgrid_plot = tgrid[plot_index]

        #read wavepacket from file
        file_wavepacket      =  params['job_directory'] + params['wavepacket_file'] + helicity + "_" + str(irun) + ".dat"

        for itime, t in zip(plot_index,list(tgrid_plot)):

            psi                  = read_wavepacket(file_wavepacket, itime, params['Nbas_global'])

            print(  "Generating plot at time = " + str('{:6.2f}'.format(t/time_to_au) ) +\
                    " " + str( self.params['time_units']) + " ----- " +\
                    "time index = " + str(itime) )
       
            rhodir = self.rho2D_calc(   psi, 
                                        polargrid, 
                                        self.params['chilist'],
                                        funcpars,  
                                        self.params['Nbas_chi'], 
                                        self.params['bound_lmax'])

            if funcpars['plot'][0] == True:

                for elem in rhodir.items():

                    plane       = elem[0]
                    rho         = elem[1]

                    funcpars['plane_split'] = self.split_word(plane)

                    # call plotting function
                    self.rho2D_plot(funcpars,polargrid,rho)


            if funcpars['save'] == True:

                with open( params['job_directory'] +  "rho2D" + "_"+ helicity + ".dat" , 'w') as rhofile:   
                    np.savetxt(rhofile, rho2D, fmt = '%10.4e')

    def rho2D_plot(self,funcpars,polargrid,rho): 
        """ Produces contour plot for 2D spatial electron density f = rho(r,theta) """

        plot_params = funcpars['plot'][1] #all plot params
        rtuple      = funcpars['rtulpe'] #range for r
        thtuple     = funcpars['thtuple'] #range for theta

        """
        Args:
            polargrid: np.array of size (nptsr,nptsth,2): (r,theta) coordinates on a meshgrid
            rho: array of size (nptsr,nptsth): function values at each point of the polar meshgrid        Comments:
            plot_params: parameters of the plot loaded from GRAPHICS.py
        """

        figsizex    = plot_params['figsize_x'] #size of the figure on screen
        figsizey    = plot_params['figsize_y']  #size of the figure on screen
        resolution  = plot_params['resolution']  #resolution in dpi

        fig         = plt.figure(figsize=(figsizex, figsizey), dpi=resolution,
                        constrained_layout=True)
        grid_fig    = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

        ax1         = fig.add_subplot(grid_fig[0, 0], projection='polar')

        cmap = matplotlib.cm.jet #jet, cool, etc
        norm = matplotlib.colors.Normalize(vmin=plot_params['vmin'], vmax=plot_params['vmax'])

        ax1.set_xlim(thtuple[0],thtuple[1]) #theta scale
        ax1.set_ylim(rtuple[0],rtuple[1]) #radial scale

        plot_params['thticks']   = list(np.linspace(thtuple[0],thtuple[1],plot_params['nticks_th']))
        plot_params['rticks']   = list(np.linspace(rtuple[0],rtuple[1],plot_params['nticks_rad'])) 
                            

        plot_rho2D  = ax1.contourf( polargrid[1], 
                                    polargrid[0], 
                                    rho,  
                                    plot_params['ncont'], 
                                    cmap = 'jet', 
                                    vmin = plot_params['vmin'],
                                    vmax = plot_params['vmax'])
        

        ax1.set_title(  label               = plot_params['title_text'],
                        fontsize            = plot_params['title_size'],
                        color               = plot_params['title_color'],
                        verticalalignment   = plot_params['title_vertical'],
                        horizontalalignment = plot_params['title_horizontal'],
                        #position            = plot_params[ "title_position"],
                        pad                 = plot_params['title_pad'],
                        backgroundcolor     = plot_params['title_background'],
                        fontname            = plot_params['title_fontname'],
                        fontstyle           = plot_params['title_fontstyle'])

        ax1.set_xlabel( xlabel              = funcpars['plane_split'][0],
                        fontsize            = plot_params['xlabel_size'],
                        color               = plot_params['label_color'],
                        loc                 = plot_params['xlabel_loc'],
                        labelpad            = plot_params['xlabel_pad'] )

        ax1.set_ylabel( ylabel              = funcpars['plane_split'][1],
                        labelpad            = plot_params['ylabel_pad'],
                        loc                 = 'center',
                        rotation            = 0)

        ax1.yaxis.grid(linewidth=0.5,alpha=0.7,color = '0.8',visible=True)
        ax1.xaxis.grid(linewidth=0.5,alpha=0.7, color = '0.8',visible=True)


        #ax1.set_xticks(plot_params['thticks']) #positions of th-ticks
        #ax1.set_yticks(plot_params['rticks']) #positions of r-ticks

        #ax1.set_xticklabels(plot_params['thticks'],fontsize=8) #th-ticks labels
        #ax1.set_yticklabels(plot_params['rticks'],fontsize=8) #r-ticks labels

        #ax1.xaxis.set_major_formatter(FormatStrFormatter(plot_params['xlabel_format'])) #set tick label formatter 
        #ax1.yaxis.set_major_formatter(FormatStrFormatter(plot_params['ylabel_format']))

        """fig.colorbar(   mappable            = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax                  = ax1, 
                        orientation         = plot_params['cbar_orientation'],
                        label               = plot_params['cbar_label'],
                        fraction            = plot_params['cbar_fraction'],
                        aspect              = plot_params['cbar_aspect'],
                        shrink              = plot_params['cbar_shrink'],
                        pad                 = plot_params['cbar_pad'],
                        panchor             = plot_params['cbar_panchor'],
                        extend              = plot_params['cbar_extend'],
                        ticks               = plot_params['cbar_ticks'],
                        drawedges           = plot_params['cbar_drawedges'],
                        format              = plot_params['cbar_format'])
        """
        if plot_params['save'] == True:
            fig.savefig(    fname       = plot_params['save_name'],
                            dpi         = plot_params['save_dpi'],
                            orientation = plot_params['save_orientation'],
                            bbox_inches = plot_params['save_bbox_inches'],
                            pad_inches  = plot_params['save_pad_inches']
                            )

        #ax1.legend() #show legends
        plt.show()
        plt.close()



    def rho2D_calc(self, psi, polargrid, chilist, funcpar, Nbas_chi,  lmax):

        plane       = funcpar['plane']
        coeff_thr   = funcpar['coeff_thr']


        rhodir = {} #dictionary containing numpy arrays representing the electron density in (r,theta) meshgrid

        for elem in plane:

            wfn = np.zeros((polargrid[0].shape[0],polargrid[1].shape[1]), dtype=complex)
            aux = np.zeros((polargrid[0].shape[0],polargrid[1].shape[1]), dtype=complex)

            if elem == "XY":
                print("Evaluation plane for rho2D: " + elem)
                theta0 = np.pi / 2.0
                #calculate spherical harmonics on the angular grid for all quantum numbers
                Ymat = np.zeros((lmax+1,2*lmax+1,polargrid[0].shape[0],polargrid[1].shape[1]),dtype=complex)
                
                for l in range(lmax+1):
                    for m in range(-l,l+1):
                        Ymat[l,l+m,:,:] =  spharm(l, m, theta0, polargrid[1]) 
                
                icoeff = 0
                
                for xi in range(Nbas_chi):
                    chi_rgrid = chilist[xi](polargrid[0])
                    aux = 0.0
                    for l in range(lmax+1):
                        for m in range(-l,l+1):
                            if np.abs((psi[2*icoeff] + 1j*psi[2*icoeff+1])) > coeff_thr:
                                aux += (psi[2*icoeff] + 1j*psi[2*icoeff+1])  * Ymat[l,l+m,:,:]
                            icoeff += 1
                    wfn += chi_rgrid * aux
                rhodir['XY'] = np.abs(wfn)**2/np.max(np.abs(wfn)**2)

            elif elem == "XZ":
                print("Evaluation plane for rho2D: " + elem)
                phi0 =  0.0

                #calculate spherical harmonics on the angular grid for all quantum numbers
                Ymat = np.zeros((lmax+1,2*lmax+1,polargrid[0].shape[0],polargrid[1].shape[1]),dtype=complex)
                
                for l in range(lmax+1):
                    for m in range(-l,l+1):
                        Ymat[l,l+m,:,:] =  spharm(l, m, polargrid[1], phi0) 

                icoeff = 0
                
                for xi in range(Nbas_chi):
                    chi_rgrid = chilist[xi](polargrid[0])
                    aux = 0.0
                    for l in range(lmax+1):
                        for m in range(-l,l+1):
                            if np.abs((psi[2*icoeff] + 1j*psi[2*icoeff+1])) > coeff_thr:
                                aux += (psi[2*icoeff] + 1j*psi[2*icoeff+1])  * Ymat[l,l+m,:,:]
                            icoeff += 1
                    wfn += chi_rgrid * aux
                

                rhodir['XZ'] = np.abs(wfn)**2/np.max(np.abs(wfn)**2)


            elif elem == "YZ":
                print("Evaluation plane for rho2D: " + elem)
                phi0 =  np.pi/2
             
                #calculate spherical harmonics on the angular grid for all quantum numbers
                Ymat = np.zeros((lmax+1,2*lmax+1,polargrid[0].shape[0],polargrid[1].shape[1]),dtype=complex)
                
                for l in range(lmax+1):
                    for m in range(-l,l+1):
                        Ymat[l,l+m,:,:] =  spharm(l, m, polargrid[1], phi0) 

                icoeff = 0
                
                for xi in range(Nbas_chi):
                    chi_rgrid = chilist[xi](polargrid[0])
                    aux = 0.0
                    for l in range(lmax+1):
                        for m in range(-l,l+1):
                            if np.abs((psi[2*icoeff] + 1j*psi[2*icoeff+1])) > coeff_thr:
                                aux += (psi[2*icoeff] + 1j*psi[2*icoeff+1])  * Ymat[l,l+m,:,:]
                            icoeff += 1
                    wfn += chi_rgrid * aux
                

                rhodir['YZ'] = np.abs(wfn)**2/np.max(np.abs(wfn)**2)

            elif len(elem) == 3:
                print("Evaluation plane defined by normal vector: " + str(elem))
                raise NotImplementedError("Feature not yet implemented")

            else:
                raise ValueError("incorrect evaluation plane for rho2D")

        

        return rhodir




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

    """ generate maps and grids """
    params['maparray_global'], params['Nbas_global']    = MAPPING.GENMAP_FEMLIST(   params['FEMLIST'],
                                                                                    params['bound_lmax'],
                                                                                    params['map_type'],
                                                                                    params['job_directory'] )

    params['maparray_chi'], params['Nbas_chi']          = MAPPING.GENMAP_FEMLIST(   params['FEMLIST'],  
                                                                                    0,
                                                                                    params['map_type'], 
                                                                                    path )

    params['Gr'], params['Nr ']                         = GRID.r_grid(              params['bound_nlobs'], 
                                                                                    params['bound_nbins'] , 
                                                                                    params['bound_binw'],  
                                                                                    params['bound_rshift'] )

    params['Gr_prim'], params['Nr_prim']                = GRID.r_grid_prim(         params['bound_nlobs'], 
                                                                                    params['bound_nbins'], 
                                                                                    params['bound_binw'], 
                                                                                    params['bound_rshift'] )

    params['chilist']                                   = PLOTS.interpolate_chi(    params['Gr_prim'], 
                                                                                    params['bound_nlobs'], 
                                                                                    params['bound_nbins'], 
                                                                                    params['bound_binw'], 
                                                                                    params['maparray_chi'])

    """ Read grid of Euler angles"""
    with open( "grid_euler.dat" , 'r') as eulerfile:   
        grid_euler = np.loadtxt(eulerfile)

    grid_euler = grid_euler.reshape(-1,3)

    N_Euler = grid_euler.shape[0]

    N_per_batch = int(N_Euler/params['N_batches'])


    params['helicity'] = pull_helicity(params)





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

        #if params['density_averaging'] == True:
        #    print( "Rotational density at point " + str([alpha, beta, gamma]) + " is: " + str(rho[irun]))

        params['irun'] = irun 

        analysis_obj = analysis(params)
        
        for elem in params['analyze_functions']:

            #call function by name given in the dictionary
            func = getattr(analysis_obj,elem['name'])
            print("Calling analysis function: " + str(elem['name']))
            func(elem)
        
            """ calculate contribution to averaged quantities"""




        """
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
    """