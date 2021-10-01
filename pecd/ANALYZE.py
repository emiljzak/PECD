from logging import warning
import numpy as np
import json
from scipy.special import sph_harm
import h5py

import time
import os
import sys
import warnings

import MAPPING
import GRID
import CONSTANTS
import PLOTS

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

from pyhank import qdht, iqdht, HankelTransform

class analysis:

    def __init__(self,params):
        self.params = params


    def spharm(self,l,m,theta,phi):
        return sph_harm(m, l, phi, theta)
        


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


    def setup_timegrids(self, momentum_analyze_times):
        tgrid,dt            = self.calc_tgrid()

        tgrid_plot_index    =  self.calc_plot_times(tgrid,dt,momentum_analyze_times) #plot time indices

        tgrid_plot          = tgrid[tgrid_plot_index]

        return tgrid_plot, tgrid_plot_index

    def split_word(self,word):
        return [char for char in word]
      
  
    def read_wavepacket(self, filename, plot_index, tgrid_plot, Nbas):

        if self.params['wavepacket_format'] == "dat":
            coeffs = []
            start_time = time.time()

            with open(filename, 'r', ) as f:
                for elem in plot_index:
                    for _ in range(elem):
                        next(f)
                    for line in f:
                        words   = line.split()
                        for ivec in range(2*Nbas):
                            coeffs.append(float(words[1+ivec]))
            warnings.warn(".dat format has been deprecated. Errors are likely to follow.")

            end_time = time.time()

            print("time for reading .dat wavepacket file =  " + str("%10.3f"%(end_time - start_time)) + "s")

        elif self.params['wavepacket_format'] == "h5":
            start_time = time.time()

            wavepacket = np.zeros( (tgrid_plot.shape[0],Nbas), dtype=complex)
            print(wavepacket.shape)
            with h5py.File(filename, 'r') as h5:
                for i,(ind,t) in enumerate(zip(plot_index,list(tgrid_plot))):
                    wavepacket[i,:] = h5[str('{:10.3f}'.format(t))][:]
            end_time = time.time()
            print("time for reading .h5 wavepacket file =  " + str("%10.3f"%(end_time - start_time)) + "s")

        return wavepacket

    def pull_helicity(self):
        if self.params['field_type']['function_name'] == "fieldRCPL":
            helicity = "R"
        elif self.params['field_type']['function_name'] == "fieldLCPL":
            helicity = "L"  
        elif self.params['field_type']['function_name'] == "fieldLP":
            helicity = "0"
        else:
            raise ValueError("Incorect field name")
        return helicity
        

 

class spacefuncs(analysis):
    
    """ Class of real space functions"""

    def __init__(self,params):
        self.params = params

    def rho2D(self,funcpars):
        print("Calculating 2D electron density")
        
        irun = self.params['irun']

        helicity  = self.pull_helicity()
        params['helicity'] = helicity

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

        plot_index = self.calc_plot_times(tgrid,dt,self.params['space_analyze_times']) #plot time indices

        tgrid_plot = tgrid[plot_index]

        #read wavepacket from file

        if params['wavepacket_format'] == "dat":
            file_wavepacket  =  self.params['job_directory'] + self.params['wavepacket_file'] + helicity + "_" + str(irun) + ".dat"
        
        elif params['wavepacket_format'] == "h5":
            file_wavepacket  =  self.params['job_directory'] + self.params['wavepacket_file'] + helicity + "_" + str(irun) + ".h5"

        #we pull the wavepacket at times specified in tgrid_plot and store it in wavepacket array
        wavepacket           = self.read_wavepacket(file_wavepacket, plot_index, tgrid_plot, self.params['Nbas_global'])

        for i, (itime, t) in enumerate(zip(plot_index,list(tgrid_plot))):
  
            print(  "Generating rho2D at time = " + str('{:6.2f}'.format(t/time_to_au) ) +\
                " " + str( self.params['time_units']) + " ----- " +\
                "time index = " + str(itime) )

            funcpars['t'] = t

            rhodir = self.rho2D_calc(   wavepacket[i,:], 
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

                with open( params['job_directory'] +  "rho2D" + "_" + str('{:.1f}'.format(t/time_to_au) ) +\
                    "_" + helicity + ".dat" , 'w') as rhofile:   
                    np.savetxt(rhofile, rho, fmt = '%10.4e')


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


        ax1.set_ylim(rtuple[0],rtuple[1]) #radial scale
        ax1.set_thetamin(thtuple[0]*180.0/np.pi)
        ax1.set_thetamax(thtuple[1]*180.0/np.pi)


        plot_params['thticks']  = list(np.linspace(thtuple[0],thtuple[1],plot_params['nticks_th']))
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

        ax1.xaxis.set_label_position('top') 
    
        ax1.set_xlabel( xlabel              = funcpars['plane_split'][1],
                        fontsize            = plot_params['xlabel_size'],
                        color               = plot_params['label_color'],
                        loc                 = plot_params['xlabel_loc'],
                        labelpad            = plot_params['xlabel_pad'] )

        ax1.set_ylabel( ylabel              = funcpars['plane_split'][0],
                        color               = plot_params['label_color'],
                        labelpad            = plot_params['ylabel_pad'],
                        loc                 = plot_params['ylabel_loc'],
                        rotation            = 0)

        ax1.yaxis.grid(linewidth=0.5, alpha=0.5, color = '0.8', visible=True)
        ax1.xaxis.grid(linewidth=0.5, alpha=0.5, color = '0.8',     visible=True)

        #custom ticks and labels:
        #ax1.set_xticks(plot_params['thticks']) #positions of th-ticks
        #ax1.set_yticks(plot_params['rticks']) #positions of r-ticks

        #ax1.set_xticklabels(plot_params['thticks'],fontsize=8) #th-ticks labels
        #ax1.set_yticklabels(plot_params['rticks'],fontsize=8) #r-ticks labels

        #ax1.xaxis.set_major_formatter(FormatStrFormatter(plot_params['xlabel_format'])) #set tick label formatter 
        #ax1.yaxis.set_major_formatter(FormatStrFormatter(plot_params['ylabel_format']))

        fig.colorbar(   mappable=  matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax                  = ax1, 
                        orientation         = plot_params['cbar_orientation'],
                        label               = plot_params['cbar_label'],
                        fraction            = plot_params['cbar_fraction'],
                        aspect              = plot_params['cbar_aspect'],
                        shrink              = plot_params['cbar_shrink'],
                        pad                 = plot_params['cbar_pad'],
                        extend              = plot_params['cbar_extend'],
                        ticks               = plot_params['cbar_ticks'],
                        drawedges           = plot_params['cbar_drawedges'],
                        format              = plot_params['cbar_format']
                       )
        
        if plot_params['save'] == True:

            fig.savefig(    fname       =   params['job_directory']  + "/graphics/space/"+
                                            plot_params['save_name'] + "_" +
                                            funcpars['plane_split'][1]+
                                            funcpars['plane_split'][0]+ "_" +
                                            str('{:.1f}'.format(funcpars['t']/time_to_au) ) +
                                            "_" +
                                            params['helicity'] +
                                            ".pdf",
                                            
                            dpi         =   plot_params['save_dpi'],
                            orientation =   plot_params['save_orientation'],
                            bbox_inches =   plot_params['save_bbox_inches'],
                            pad_inches  =   plot_params['save_pad_inches']
                            )

        plt.show()
        plt.close()

    def calc_spharm_array(self,lmax,plane,grid):
        """
            Returns array of spherical harmonics in range 0,...,l and m = -l, ..., l and on 
            the grid, which can be 2D or 3D meshgrid, depending on the plane chosen.
            grid[0] = r
            grid[1] = theta
            grid[2] = phi
            in the form of meshgrid
        """

        if plane == "XY":
            theta0 = np.pi / 2.0
            #calculate spherical harmonics on the angular grid for all quantum numbers
            Ymat = np.zeros((lmax+1,2*lmax+1,grid[0].shape[0],grid[1].shape[1]),dtype=complex)
            
            for l in range(lmax+1):
                for m in range(-l,l+1):
                    Ymat[l,l+m,:,:] =  self.spharm(l, m, theta0, grid[1]) 

            return Ymat

        elif plane == "XZ":

            phi0 =  0.0

            #calculate spherical harmonics on the angular grid for all quantum numbers
            Ymat = np.zeros((lmax+1,2*lmax+1,grid[0].shape[0],grid[1].shape[1]),dtype=complex)
            
            for l in range(lmax+1):
                for m in range(-l,l+1):
                    Ymat[l,l+m,:,:] =  self.spharm(l, m, grid[1], phi0) 

            return Ymat

        elif plane == "YZ":
            phi0 =  np.pi/2
            
            #calculate spherical harmonics on the angular grid for all quantum numbers
            Ymat = np.zeros((lmax+1,2*lmax+1,grid[0].shape[0],grid[1].shape[1]),dtype=complex)
            
            for l in range(lmax+1):
                for m in range(-l,l+1):
                    Ymat[l,l+m,:,:] =  self.spharm(l, m, grid[1], phi0) 

            return Ymat

        elif len(plane) == 3:
            print("Evaluation plane defined by normal vector: " + str(plane))
            raise NotImplementedError("Feature not yet implemented")

        elif plane == "XYZ":
            #calculating Ylm on full 3D (r,theta,phi) meshgrid
            
            #calculate spherical harmonics on the angular grid for all quantum numbers
            Ymat = np.zeros((lmax+1,2*lmax+1,grid[0].shape[0],grid[1].shape[1],grid[2].shape[2]),dtype=complex)
            
            for l in range(lmax+1):
                for m in range(-l,l+1):
                    Ymat[l,l+m,:,:,:] =  self.spharm(l, m, grid[1], grid[2]) 

            return Ymat

    def calc_wfn_array(self,lmax,Nbas_chi,polargrid,chilist,Ymat,coeff_thr,psi):
                            
        wfn = np.zeros((polargrid[0].shape[0],polargrid[1].shape[1]), dtype=complex)
        aux = np.zeros((polargrid[0].shape[0],polargrid[1].shape[1]), dtype=complex)

        icoeff = 0
        for xi in range(Nbas_chi):
            chi_rgrid = chilist[xi](polargrid[0])
            aux = 0.0
            for l in range(lmax+1):
                for m in range(-l,l+1):
                    if np.abs((psi[icoeff])) > coeff_thr:               
                    #if np.abs((psi[2*icoeff] + 1j*psi[2*icoeff+1])) > coeff_thr:
                        aux += (psi[icoeff])  * Ymat[l,l+m,:,:]
                        #aux += (psi[2*icoeff] + 1j*psi[2*icoeff+1])  * Ymat[l,l+m,:,:]
                    icoeff += 1
            wfn += chi_rgrid * aux
        return wfn

    def rho2D_calc(self, psi, polargrid, chilist, funcpar, Nbas_chi,  lmax):

        plane       = funcpar['plane']
        coeff_thr   = funcpar['coeff_thr']

        rhodir = {} #dictionary containing numpy arrays representing the electron density in (r,theta) meshgrid

        for elem in plane:

            print("Evaluation plane for rho2D: " + elem)

            Ymat    = self.calc_spharm_array(lmax,elem,polargrid)
            wfn     = self.calc_wfn_array(lmax,Nbas_chi,polargrid,chilist,Ymat,coeff_thr,psi)
            rhodir[elem] = np.abs(wfn)**2/np.max(np.abs(wfn)**2)
        

        return rhodir




class momentumfuncs(analysis):

    def __init__(self,params):
        self.params = params

    def calc_grid_for_FT(self):
        """ Calculate real-space grid (r,theta) for evaluation of Hankel transform and for plottting"""
        """ The real-space grid determines the k-space grid returned by PyHank """

        nbins   = self.params['bound_nbins'] 
        rmax    = nbins * self.params['bound_binw']
        npts    = self.params['npts_r_ft'] 
        N_red   = npts 

        grid_theta  = np.linspace(-np.pi, np.pi, N_red , endpoint = False ) # 
        grid_r      = np.linspace(0.0, rmax, npts, endpoint = False)

        return grid_theta, grid_r



    def calc_wfnft(self):

        irun                    = self.params['irun']
        helicity                = self.pull_helicity()
        self.params['helicity'] = helicity

        #which grid point corresponds to the radial cut-off?
        self.params['ipoint_cutoff'] = np.argmin(np.abs(self.params['Gr'].ravel() - self.params['rcutoff']))
        print("ipoint_cutoff = " + str(self.params['ipoint_cutoff']))

        tgrid_plot, plot_index = self.params['tgrid_plot_momentum'], self.params['tgrid_plot_index_momentum'] 
        #read wavepacket from file

        if params['wavepacket_format'] == "dat":
            file_wavepacket  =  self.params['job_directory'] + self.params['wavepacket_file'] + helicity + "_" + str(irun) + ".dat"
        
        elif params['wavepacket_format'] == "h5":
            file_wavepacket  =  self.params['job_directory'] + self.params['wavepacket_file'] + helicity + "_" + str(irun) + ".h5"

        #we pull the wavepacket at times specified in tgrid_plot and store it in wavepacket array
        wavepacket           = self.read_wavepacket(file_wavepacket, plot_index, tgrid_plot, self.params['Nbas_global'])



        if self.params['FT_method']    == "FFT_cart":

            self.calc_fftcart_psi_3d(   params, 
                                        self.params['maparray_global'], 
                                        self.params['Gr'], 
                                        wavepacket,
                                        self.params['chilist'])

        elif self.params['FT_method']  == "FFT_hankel":

            grid_theta, grid_r = self.calc_grid_for_FT()


            #calculate partial waves on radial grid
            Plm         = self.calc_partial_waves(      grid_r,
                                                        wavepacket)

            #return Hankel transforms on the appropriate k-vector grid identical to grid_r
            Flm, kgrid  = self.calc_hankel_transforms(  Plm, 
                                                        grid_r)
           
        return Flm, kgrid
    


    def calc_FT_3D_hankel(self, Flm, kgrid, grid_theta, grid_r, phi0 = 0.0 ):
        """ returns: fourier transform inside a ball grid (r,theta,phi) """

        npts      = grid_r.size

        FT = np.zeros((npts, npts), dtype = complex)

        for i in range(npts ):
            #print(i)
            for elem in Flm:
                FT[i,:] +=   ((-1.0 * 1j)**elem[0]) * elem[2][:npts] * PLOTS.spharm(elem[0], elem[1], grid_theta[i] , phi0) 

        return FT, kgrid


    def calc_partial_waves(self, grid_r, wavepacket):
        """
        returns: list of numpy arrays. List is labelled by l,m and t_i from 'momentum_analyze_time'
        """

        Plm = []
        lmax            = self.params['bound_lmax']
        maparray_global = self.params['maparray_global']
        maparray_chi    = self.params['maparray_chi']
        chilist         = self.params['chilist']
        ipoint_cutoff   = self.params['ipoint_cutoff']
        
        Nt              = wavepacket.shape[0] #number of evaluation times
        Nbas            = len(maparray_global)
        Nr              = len(maparray_chi)
        npts = grid_r.size #number of radial grid point at which Plm are evaluated. This grid determines the maximum photoelectron momentum.
        

        val = np.zeros(npts, dtype = complex)


        for itime in range(Nt):
            psi = wavepacket[itime,:]
            c_arr = psi.reshape(len(maparray_chi),-1)
            print("time-point: " + str(itime))
            val = 0.0 + 1j * 0.0
            indang = 0
            for l in range(0,lmax+1):
                for m in range(-l,l+1):
                    print("P_lm: " + str(l) + " " + str(m))
                
                    for ielem, elem in enumerate(maparray_chi):
                        if elem[2] > ipoint_cutoff: #cut-out bound-state electron density

                            val +=  c_arr[ielem][indang] * chilist[elem[2]-1](grid_r)

                    indang += 1
                    Plm.append([itime,l,m,val])
                    val = 0.0 + 1j * 0.0

            if self.params['plot_Plm'] == True:


                #for i in range(Nr):
                #    plt.plot(grid_r,chilist[i](grid_r))
                #plt.show()
                #only for itime =0:
                for s in range((lmax+1)**2):
                    plt.plot(grid_r,np.abs(Plm[s][3]),marker='.',label="P_"+str(s))
                    plt.legend()
                plt.show()
                plt.close()
        return Plm


    def calc_hankel_transforms(self, Plm, grid_r):
        Flm = [] #list of output Hankel transforms
    
        for ielem, elem in enumerate(Plm):
            print("Calculating Hankel transform for time-point =  " + str(elem[0]) + " for partial wave Plm: " + str(elem[1]) + " " + str(elem[2]))

            Hank_obj = HankelTransform(elem[1], radial_grid = grid_r) #max_radius=200.0, n_points=1000) #radial_grid=fine_grid)
            #Hank.append(Hank_obj) 
            Plm_resampled = Hank_obj.to_transform_r(elem[3])
            F = Hank_obj.qdht(Plm_resampled)
            Flm.append([elem[0],elem[1],elem[2],F])
            if  params['plot_Flm']  == True:
                plt.plot(Hank_obj.kr,np.abs(F))
        
        if  params['plot_Flm']  == True:   
            plt.show()
            plt.close()

        return Flm, Hank_obj.kr


    def W2D(self,funcpars,Flm,kgrid):
        print("Calculating 2D electron momentum probability density")
        

        irun = self.params['irun']
        helicity  = self.pull_helicity()
        params['helicity'] = helicity

        """ set up 1D grids """

        if funcpars['k_grid']['type'] == "manual":
            ktuple  = (funcpars['k_grid']['kmin'], funcpars['k_grid']['kmax'], funcpars['k_grid']['npts'])
            funcpars['ktulpe'] = ktuple
            kgrid1D         = np.linspace(ktuple[0], ktuple[1], ktuple[2], endpoint=True, dtype=float)
        elif funcpars['k_grid']['type'] == "automatic":
            kgrid1D = kgrid


        thtuple             = funcpars['th_grid']
        funcpars['thtuple'] = thtuple
        unity_vec           = np.linspace(0.0, 1.0, thtuple[2], endpoint=True, dtype=float)
        thgrid1D            = thtuple[1] * unity_vec
        

        """ generate 2D meshgrid for storing W2D """
        polargrid = np.meshgrid(kgrid1D, thgrid1D)

        tgrid_plot, plot_index = self.params['tgrid_plot_momentum'], self.params['tgrid_plot_index_momentum'] 

        for i, (itime, t) in enumerate(zip(plot_index,list(tgrid_plot))):
  
            print(  "Generating W2D at time = " + str('{:6.2f}'.format(t/time_to_au) ) +\
                " " + str( self.params['time_units']) + " ----- " +\
                "time index = " + str(itime) )

            self.params['t'] = t

            Flm_t = Flm[i*(self.params['bound_lmax']+1)**2:(i+1)*(self.params['bound_lmax']+1)**2 ]
            print(Flm_t)


            W2Ddir = self.W2D_calc(funcpars,Flm,kgrid1D,thgrid1D)


            if funcpars['plot'][0] == True:

                for elem in W2Ddir.items():

                    plane       = elem[0]
                    W           = elem[1]

                    funcpars['plane_split'] = self.split_word(plane)

                    # call plotting function
                    self.W2D_plot(funcpars,polargrid,W)


            if funcpars['save'] == True:

                with open( params['job_directory'] +  "W2D" + "_" + str('{:.1f}'.format(t/time_to_au) ) +\
                    "_" + helicity + ".dat" , 'w') as rhofile:   
                    np.savetxt(rhofile, rho, fmt = '%10.4e')


        
    def W2D_calc(self,funcpar, Flm, kgrid, thgrid):
        """calculate numerically W2D for fixed phi angle"""

        plane       = funcpar['plane']

        W2Ddir = {} #dictionary containing numpy arrays representing the electron's momentum density in (r,theta) meshgrid

        for elem in plane:

            print("Evaluation plane for W2D: " + elem)

            Ymat    = self.calc_spharm_array(lmax,elem,polargrid)
            wfn     = self.calc_wfn_array(lmax,Nbas_chi,polargrid,chilist,Ymat,coeff_thr,psi)

            W2D, kgrid = self.calc_FT_3D_hankel(Plm, Flm, kgrid, params['bound_lmax'], grid_theta, grid_r, maparray_chi, maparray_global, psi, chilist, phigrid[iphi] )

:           W2Ddir[elem] = np.abs(W2D)**2/np.max(np.abs(W2D)**2)


        return 

    def calc_fftcart_psi_3d(self,params, maparray, Gr, psi, chilist):
        coeff_thr = 1e-3
        ncontours = 20

        nlobs   = params['bound_nlobs']
        nbins   = params['bound_nbins'] 
        npoints = 100
        rmax    = nbins * params['bound_binw']
        rmin    = 0.0

        fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
        spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
        axft = fig.add_subplot(spec[0, 0])

        cart_grid = np.linspace(-1.0 * rmax/np.sqrt(3), rmax/np.sqrt(3), npoints, endpoint=True, dtype=float)

        x3d, y3d, z3d = np.meshgrid(cart_grid, cart_grid, cart_grid)


        x   =  np.zeros(nlobs)
        w   =  np.zeros(nlobs)
        x,w =  GRID.gauss_lobatto(nlobs,14)
        w   =  np.array(w)
        x   =  np.array(x) # convert back to np arrays

        val = np.zeros((len(x3d),len(y3d),len(z3d)), dtype = complex)
        
        start_time = time.time()
        for ielem, elem in enumerate(maparray):
            if np.abs(psi[2*ielem]+1j*psi[2*ielem + 1]) > coeff_thr:
                print(str(elem) + str(psi[ielem]))

                for i in range(len(cart_grid)):
                    for j in range(len(cart_grid)):
                        val[i,j,:] +=  ( psi[2*ielem] + 1j * psi[2*ielem + 1] ) * spharmcart(elem[3], elem[4], x3d[i,j,:], y3d[i,j,:], z3d[i,j,:]) * \
                                        chilist[elem[2]-1](np.sqrt(x3d[i,j,:]**2 + y3d[i,j,:]**2 + z3d[i,j,:]**2)) #


        end_time = time.time()
        print("The time calculation of wavefunction on 3-D cubic grid: " + str("%10.3f"%(end_time-start_time)) + "s")

        start_time = time.time()
        ftval = fftn(val)
        end_time = time.time()
        print("The time calculation 3D Fourier transform: " + str("%10.3f"%(end_time-start_time)) + "s")

        print(np.shape(ftval))

        ft_grid = np.linspace(-1.0/(rmax), 1.0/(rmax), npoints, endpoint=True, dtype=float)

        yftgrid, zftgrid = np.meshgrid(ft_grid,ft_grid)

        line_ft = axft.contourf(yftgrid, zftgrid , ftval[50,:npoints,:npoints].real/np.max(np.abs(ftval)), 
                                            ncontours, cmap = 'jet', vmin=-0.2, vmax=0.2) #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
        plt.colorbar(line_ft, ax=axft, aspect=30)
        
        #axradang_r.set_yticklabels(list(str(np.linspace(rmin,rmax,5.0)))) # set radial tick label
        plt.legend()   
        plt.show()  


class averagedobs:
    def __init__(self):
        pass


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

        analysis_obj    = analysis(params)
        

        params['tgrid_plot_space'], params['tgrid_plot_index_space'] = analysis_obj.setup_timegrids(params['space_analyze_times'])


        spaceobs        = spacefuncs(params)


        for elem in params['analyze_space']:
            # Future note: we may want to pull out the wavefunction calculation into a general routine
            # independent of the called function. This is going to save some time.

            #call function by name given in the dictionary
            func = getattr(spaceobs,elem['name'])
            print("Calling space function: " + str(elem['name']))
            func(elem)
        


        """ set up time grids for evaluating wfn """

        params['tgrid_plot_momentum'], params['tgrid_plot_index_momentum'] = analysis_obj.setup_timegrids(params['momentum_analyze_times'])

        momentumobs     = momentumfuncs(params)

        Flm, kgrid = momentumobs.calc_wfnft()

        for elem in params['analyze_momentum']:

            func = getattr(momentumobs,elem['name'])
            print("Calling momentum function: " + str(elem['name']))
            func(elem, Flm, kgrid)
        


            """ calculate contribution to averaged quantities"""

            """
                        if params['density_averaging'] == True:
                #plot_W_3D_num(params, maparray_chi, maparray_global, psi, chilist, gamma)
                Wav += float(rho[irun]) * np.abs(FT)**2
                #PLOTS.plot_2D_polar_map(np.abs(FT)**2,grid_theta,kgrid,100)

            else:
                print("proceeding with uniform rotational density")
                Wav += np.abs(FT)**2

            
            """
