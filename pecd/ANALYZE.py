from ctypes import pointer
from logging import warning
import numpy as np
import json

from numpy.core.fromnumeric import size
from scipy.special import sph_harm
from scipy.special import eval_legendre
from scipy import integrate
from scipy import interpolate   


import h5py

import time
import os
import sys
import warnings

import MAPPING
import GRID
import CONSTANTS
import PLOTS
import GRAPHICS

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

from pyhank import qdht, iqdht, HankelTransform

class analysis:

    def __init__(self,params):
        self.params = params

    def find_nearest(self,array, value):
        array   = np.asarray(array)
        idx     = (np.abs(array - value)).argmin()
        return array[idx], idx


    def gen_meshgrid_2D(self,xrange,yrange,nptsx,nptsy):
        """
        nptsx,nptsy: int: number of sampling points along x,y
        """
        x1d = np.linspace(xrange[0], xrange[1], nptsx, endpoint=True, dtype=float)
        y1d = np.linspace(yrange[0], yrange[1], nptsy, endpoint=True, dtype=float)

        x2d,y2d = np.meshgrid(x1d,y1d,indexing='ij')
        return x2d,y2d



    def eval_func_meshgrid_2D(x2d,y2d,func):

        v2d = eval(func+"(x2d,y2d)") #dynamic call
        #v2d = tanh2d(x2d,y2d) #static call
        return v2d

    def spharm(self,l,m,theta,phi):
        return sph_harm(m, l, phi, theta)
    
    def rotate_spharm(self,func,D):
        rot_angle = [0.0, np.pi/4, 0.0]
        func_rotated = np.zeros((func.shape[0], func.shape[1]))
        for m in range(D.shape[1]):
            func_rotated += D[:,m] * func[:,m]

        return func_rotated



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
                    #wavepacket[i,:] = h5[str('{:10.3f}'.format(t))][:]
                    #print("Keys: %s" % h5.keys())
                    #a_group_key = list(h5.keys())[0]

                    # Get the data
                    #data = list(h5[a_group_key])
                    #print(data)
                    #exit()
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
        

    def calc_spharm_array(self,lmax,plane,grid,phi=None):
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
            Ymat = np.zeros((lmax+1, 2*lmax+1, grid[0].shape[0], grid[1].shape[1]), dtype=complex)
            
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
 
        elif plane == "Z":
            # calculate Ylm for arbitrary phi0
            #calculate spherical harmonics on the angular grid for all quantum numbers
            Ymat = np.zeros((lmax+1,2*lmax+1,grid[0].shape[0],grid[1].shape[1]),dtype=complex)
            
            for l in range(lmax+1):
                for m in range(-l,l+1):
                    Ymat[l,l+m,:,:] =  self.spharm(l, m, grid[1], phi) 

            return Ymat

    def legendre_expansion(self,funcpars,grid,fdir): 
        """ Calculate the Legendre expansion coefficients as a function of the photo-electron momentum """

        for elem in fdir.items():

            f       = elem[1]
            plane   = elem[0]
            print("plane = " + str(plane))
            kgrid       = grid[0]
            thetagrid   = grid[1]

            """ Interpolate f(k,theta)"""
            #W_interp    = interpolate.interp2d(kgrid, thetagrid, f, kind='cubic')
            W_interp    = interpolate.RectBivariateSpline(kgrid[:,0], thetagrid[0,:], f[:,:], kx=3, ky=3)

            if self.params['Leg_test_interp'] == True:

                fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
                spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
                ax = fig.add_subplot(spec[0, 0], projection='polar')

                #plot W_av on the original grid
                W_interp_mesh = W_interp(kgrid[:,0], thetagrid[0,:])
              
                #line_W_original = ax.contourf(kgrid, thetagrid, W_interp_mesh, 
                #                        100, cmap = 'jet') 
                #plt.show()
                #plt.close()
                #plot W_av on test grid 
                thetagridtest       = np.linspace(0, 2.0*np.pi, 400)
                kgridtest           = np.linspace(0, 2, 600)
                kgridtestmesh ,thetatestmesh    = np.meshgrid(kgridtest, thetagridtest,indexing='ij' )
                W_interp_testmesh   = W_interp( kgridtest , thetagridtest )
                line_test = ax.contourf(  thetatestmesh ,kgridtestmesh, W_interp_testmesh , 
                                        100, cmap = 'jet') 
                plt.show()
                plt.close()


            Lmax    = self.params['Leg_lmax'] 
            nleg     = 100
            
            x, w    = np.polynomial.legendre.leggauss(nleg)


            #print(x.shape)
            #print(w.shape)
            #print(x)
            #print("legendre grid")
            #print(-np.arccos(x)+np.pi)

            #exit()
            w       = w.reshape(nleg,-1)

            nkpoints    = self.params['Leg_npts_r'] # number of radial grid points at which the b-coefficients are calculated
            bcoeff      = np.zeros((nkpoints,Lmax+1), dtype = float)

            kgrid1D       = np.linspace(0.05, self.params['pes_max_k'], nkpoints)
            thgrid1D      = np.linspace(0.0, 2.0 * np.pi, self.params['Leg_npts_th'], endpoint=False )
            
            
            """ calculating Legendre moments """
            print("Calculating b(k) coefficients in k range: " + str(0.05) + " ... " + str(self.params['pes_max_k']))
            
            #print("-acos(x) = " + str(-np.arccos(x)))
            #Pn = eval_legendre(2, x).reshape(nleg,-1) 
            #plt.plot(x,Pn)
            #plt.show()
            #exit()
            
            #kgridmesh_btest, thetamesh_btest = np.meshgrid(kgrid1D, -np.arccos(x)+np.pi,indexing='ij')
            #print("legendre grid")
            #print(-np.arccos(x)+np.pi)


            for n in range(0,Lmax+1):

                Pn = eval_legendre(n, x).reshape(nleg,-1)

                for ipoint,k in enumerate(list(kgrid1D)):

                    W_interp1        = W_interp(k,-np.arccos(x)+2.0*np.pi).reshape(nleg,-1)

                    bcoeff[ipoint,n] = np.sum(w[:,0] * W_interp1[:,0] * Pn[:,0]) * (2.0 * n + 1.0) / 2.0

                #print(W_interp1.shape)
                #W_interp2    = W_interp(kgrid1D,-np.arccos(x)+np.pi)
                #fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
                #spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
                #ax = fig.add_subplot(spec[0, 0], projection='polar')
                #line_test = ax.contourf(  thetamesh_btest , kgridmesh_btest, W_interp2, 
                #        100, cmap = 'jet')
                #plt.show()
                #exit()
                if self.params['plot_bcoeffs'] == True:
                    plt.plot(kgrid1D,np.log(bcoeff[:,n]),label=n)
                    plt.legend()

            if self.params['plot_bcoeffs'] == True:
                plt.show()
                plt.close()
        
           
            #plot Legendre-reconstructed distribution on test grid 


            
            kgrid_leg_mesh, thetagrid_leg_mesh = np.meshgrid(kgrid1D, thgrid1D, indexing='ij')

            if self.params['Leg_plot_reconst'] == True:

                W_legendre = np.zeros((kgrid_leg_mesh.shape[0],thetagrid_leg_mesh.shape[1]), dtype = float)

                for ipoint in range(nkpoints):

                    for n in range(0,Lmax+1):

                        Pn                      = eval_legendre(n, np.cos(thetagrid_leg_mesh[0,:])).reshape(thetagrid_leg_mesh.shape[1],1)
                        W_legendre[ipoint,:]    += bcoeff[ipoint,n] * Pn[:,0] 
                       
                fig     = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
                spec    = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
                ax      = fig.add_subplot(spec[0, 0], projection='polar')
                
                line_legendre = ax.contourf(    thetagrid_leg_mesh,  kgrid_leg_mesh,
                                                W_legendre / W_legendre.max(), 
                                                100, 
                                                cmap = 'jet') 

                plt.colorbar(line_legendre, ax=ax, aspect=30) 
                plt.show()

        return bcoeff, kgrid



    def calc_rgrid_for_FT(self):
        """ Calculate real-space grid (r,theta) for evaluation of Hankel transform and for plottting"""
        """ The real-space grid determines the k-space grid returned by PyHank """

        nbins   = self.params['prop_nbins'] 
        rmax    = nbins * self.params['bound_binw']
        npts    = self.params['npts_r_ft'] 
        N_red   = npts 

        grid_theta  = np.linspace(-np.pi, np.pi, N_red , endpoint = False ) # 
        grid_r      = np.linspace(0.0, rmax, npts, endpoint = False)

        return grid_theta, grid_r



    def calc_Flm(self,helicity):

        irun                    = self.params['irun']

        # which grid point corresponds to the radial cut-off?
        self.params['ipoint_cutoff'] = np.argmin(np.abs(self.params['Gr'].ravel() - self.params['rcutoff']))
        print("ipoint_cutoff = " + str(self.params['ipoint_cutoff']))

        """ set up time grids for evaluating wfn """
        tgrid_plot, plot_index   = self.setup_timegrids(self.params['momentum_analyze_times'])
        
        # read wavepacket from file
        if self.params['wavepacket_format'] == "dat":
            file_wavepacket  =  self.params['job_directory'] + self.params['wavepacket_file'] + helicity + "_" + str(irun) + ".dat"
        
        elif self.params['wavepacket_format'] == "h5":
            file_wavepacket  =  self.params['job_directory'] + self.params['wavepacket_file'] + helicity + "_" + str(irun) + ".h5"


        assert os.path.isfile(file_wavepacket)
        # we pull the wavepacket at times specified in tgrid_plot and store it in wavepacket array
        wavepacket           = self.read_wavepacket(file_wavepacket, plot_index, tgrid_plot, self.params['Nbas_global'])



        if self.params['FT_method']    == "FFT_cart":

            self.calc_fftcart_psi_3d(   self.params, 
                                        self.params['maparray_global'], 
                                        self.params['Gr'], 
                                        wavepacket,
                                        self.params['chilist'])

        elif self.params['FT_method']  == "FFT_hankel":

            grid_theta, grid_r = self.calc_rgrid_for_FT()

            #calculate partial waves on radial grid
            Plm         = self.calc_partial_waves(      grid_r,
                                                        wavepacket)

            #return Hankel transforms on the appropriate k-vector grid identical to grid_r
            Flm, kgrid  = self.calc_hankel_transforms(  Plm, 
                                                        grid_r)
           
        return Flm, kgrid
    

    def calc_FT_3D_hankel(self, Flm,  Ymat):
        """ returns: fourier transform inside a ball grid (r,theta,phi), or disk (r,theta,phi0) or disk (r,theta0,phi)  """

        npts_k       = Ymat.shape[2]
        npts_th      = Ymat.shape[3]

        n_waves      = (self.params['bound_lmax']+1)**2
        #print("size of theta grid = " + str(npts_th))
        #print("size of kgrid = " + str(npts_k))

        FT = np.zeros((npts_k, npts_th), dtype = complex)
        Flm2D = np.zeros((n_waves, npts_k, npts_th), dtype = complex)


        for ielem, elem in enumerate(Flm):

            for ith in range(npts_th):
                Flm2D[ielem,:,ith] = elem[3]

            FT +=   ((-1.0 * 1j)**elem[1]) * Flm2D[ielem,:,:] * Ymat[elem[1], elem[1]+elem[2],:,:]

        return FT


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



    def barray_plot_2D(self,grid_euler,ibcoeff,funcpars):
        """ Produces contour plot for b(beta,gamma) """

        """
        Args:
            
        Comments:
            1)

        """

        cont2D_params = funcpars['plot'][1] #all plot params

        N_Euler = grid_euler.shape[0]
    
        barray = np.zeros((N_Euler,4+2*(self.params['Leg_lmax']+1)), dtype = float)
        
        #for each time pointer
        #for each k pointer

        for irun in range(N_Euler):
            barray[irun,0],barray[irun,1] = grid_euler[irun,1], grid_euler[irun,2]
            for t in list(self.params['momentum_analyze_times']):
                file = self.params['job_directory'] +  "bcoeffs" +\
                            "_" + str(irun) + "_"  + str('{:.1f}'.format(t) ) +\
                            ".dat"

                barray[irun,2] = t
                if not os.path.isfile(file):
                    continue
                else:
                    with open(  file , 'r') as pecdfile:
                        barray[irun,2] = t
                        #for ikelem, k in enumerate(self.params['pecd_momenta']):

                        for line in pecdfile:
                            words   = line.split()
                            barray[irun,3] = float(words[0])
                            for il in range(2*(self.params['Leg_lmax']+1)):
        
                                barray[irun,il+4] = float(words[il+1])

        with open( params['job_directory'] +  "barray.dat" , 'w') as barfile:   
            np.savetxt(barfile, barray, fmt = '%12.8f')


        bcoef = barray[:,4+ibcoeff]

        cmap = matplotlib.cm.jet #jet, cool, etc

        norm = matplotlib.colors.Normalize(vmin = bcoef.min(), vmax = bcoef.max())

        figsizex = cont2D_params['figsize_x'] #size of the figure on screen
        figsizey = cont2D_params['figsize_y']  #size of the figure on screen
        resolution = cont2D_params['resolution']  #resolution in dpi

        fig = plt.figure(figsize=(figsizex, figsizey), dpi=resolution,
                        constrained_layout=True)
        grid_fig = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

        ax1 = fig.add_subplot(grid_fig[0, 0], projection='rectilinear')


        #x2d,y2d = np.meshgrid(barray[:,1],barray[:,0], 'ij')
        #bcoef_int = interpolate.intetrp2d(x2d,y2d,bcoef)

        plot_cont_1 = ax1.tricontourf( barray[:,1]/np.pi,barray[:,0]/np.pi, bcoef, 
                                    cont2D_params['ncont'], 
                                    cmap = 'jet')
        
        ax1.set_title(  label               = "",
                        fontsize            = cont2D_params['title_size'],
                        color               = cont2D_params['title_color'],
                        verticalalignment   = cont2D_params['title_vertical'],
                        horizontalalignment = cont2D_params['title_horizontal'],
                        #position            = cont2D_params[ "title_position"],
                        pad                 = cont2D_params['title_pad'],
                        backgroundcolor     = cont2D_params['title_background'],
                        fontname            = cont2D_params['title_fontname'],
                        fontstyle           = cont2D_params['title_fontstyle'])

        ax1.set_xlabel( xlabel              = cont2D_params['xlabel'],
                        fontsize            = cont2D_params['xlabel_size'],
                        color               = cont2D_params['label_color'],
                        loc                 = cont2D_params['xlabel_loc'],
                        labelpad            = cont2D_params['xlabel_pad'] )

        ax1.set_ylabel(cont2D_params['ylabel'])

    
        ax1.set_xticks(cont2D_params['xticks']) #positions of x-ticks
        ax1.set_yticks(cont2D_params['yticks']) #positions of y-ticks

        ax1.set_xticklabels(cont2D_params['xticks']) #x-ticks labels
        ax1.set_yticklabels(cont2D_params['yticks']) #y-ticks labels

        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) #set tick label formatter 
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        fig.colorbar(   mappable =  matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax                  = ax1, 
                        orientation         = cont2D_params['cbar_orientation'],
                        label               = r'$b_{0}^{{CPR}}$'.format(ibcoeff)+"(%)",
                        fraction            = cont2D_params['cbar_fraction'],
                        aspect              = cont2D_params['cbar_aspect'],
                        shrink              = cont2D_params['cbar_shrink'],
                        pad                 = cont2D_params['cbar_pad'],
                        extend              = cont2D_params['cbar_extend'],
                        ticks               = cont2D_params['cbar_ticks'],
                        drawedges           = cont2D_params['cbar_drawedges'],
                        format              = cont2D_params['cbar_format'])
        
        if cont2D_params['save'] == True:
            fig.savefig(    fname       = "b"+str(ibcoeff) + "_" +  cont2D_params['save_name'],
                            dpi         = cont2D_params['save_dpi'],
                            orientation = cont2D_params['save_orientation'],
                            bbox_inches = cont2D_params['save_bbox_inches'],
                            pad_inches  = cont2D_params['save_pad_inches']
                            )

        #ax1.legend() #show legends
        plt.show()
        plt.close()


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
                    self.rho2D_plot(funcpars,polargrid,rho,irun)


            if funcpars['save'] == True:

                with open( params['job_directory'] +  "rho2D" + "_" + str('{:.1f}'.format(t/time_to_au) ) +\
                    "_" + helicity + ".dat" , 'w') as rhofile:   
                    np.savetxt(rhofile, rho, fmt = '%10.4e')


    def rho2D_plot(self,funcpars,polargrid,rho,irun): 
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
        ax1.xaxis.grid(linewidth=0.5, alpha=0.5, color = '0.8', visible=True)


        ax1.text(   0.0, 0.0, str('{:.1f}'.format(funcpars['t']/time_to_au) ) + " as", 
                    color = plot_params['time_colour'], fontsize = plot_params['time_size'],
                    alpha = 0.5,
                    transform = ax1.transAxes)

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
                                            params['helicity'] + "_" + str(irun) +
                                            ".pdf",
                                            
                            dpi         =   plot_params['save_dpi'],
                            orientation =   plot_params['save_orientation'],
                            bbox_inches =   plot_params['save_bbox_inches'],
                            pad_inches  =   plot_params['save_pad_inches']
                            )

        if funcpars['show'] == True:

            plt.show()
            plt.close()


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


    def PECD(self,funcpars):
        
       
        print("------------ Quantitative PECD analysis --------------"+'\n\n')

        """ Calculate W2D for left- and right-circularly polarized light """

        print("Calculating Flm for left- and right-circularly polarized pulse")

        self.params['field_type']['function_name'] = "fieldRCPL"
        obs_R, polargrid = self.W2Dav(self.params['W2Dav'])

        self.params['field_type']['function_name'] = "fieldLCPL" 
        obs_L, _ = self.W2Dav(self.params['W2Dav'])
        

        """ ************** Calculate PECD **************** """
        
        """ set up time grids PECD """
        tgrid_plot, plot_index   = self.setup_timegrids(self.params['momentum_analyze_times'])
        
        for i, (itime, t) in enumerate(zip(plot_index,list(tgrid_plot))):
  
            print(  "Generating PECD at time = " + str('{:6.2f}'.format(t/time_to_au) ) +\
                " " + str( self.params['time_units']) + " ----- " +\
                "time index = " + str(itime) )

            funcpars['t'] = t


            Wav_L = obs_L[i][2]['Wav']
            Wav_R = obs_R[i][2]['Wav']

            bcoeff_L    = obs_L[i][2]['bcoeff']
            bcoeff_R    = obs_R[i][2]['bcoeff']
            kgrid       = obs_R[i][2]['kgrid']

            pecd = Wav_R-Wav_L

            # calculating differential W2Dav:
            if funcpars['plot2D'][0] == True:
                self.PECD_plot2D(funcpars,polargrid,pecd)

            k_pecd      = [] # values of electron momentum at which PECD is evaluated
            ind_kgrid   = [] # index of electron momentum in the list


        
            for kelem in params['pecd_momenta']:
                k, ind = self.find_nearest(kgrid[:,0], kelem)
                k_pecd.append(k)
                ind_kgrid.append(ind)


            pecd_list = []

            if funcpars['save'] == True:
   
                with open(  self.params['job_directory'] +  "PECD" +\
                            "_" + str(irun) + "_"  + str('{:.1f}'.format(t/time_to_au) ) +\
                            ".dat" , 'w') as pecdfile:

                    np.savetxt(pecdfile, pecd , fmt = '%10.4e')

   
                with open(  self.params['job_directory'] +  "bcoeffs" +\
                            "_" + str(irun) + "_"  + str('{:.1f}'.format(t/time_to_au) ) +\
                            ".dat" , 'w') as pecdfile:
                    for ielem, k in enumerate(k_pecd):
                        pecdfile.write(    str('{:8.2f}'.format(k)) +\
                                            " ".join('{:12.8f}'.format(bcoeff_R[ielem,n]/bcoeff_R[ielem,0]) for n in range(self.params['pecd_lmax']+1)) +\
                                            " ".join('{:12.8f}'.format(bcoeff_L[ielem,n]/bcoeff_L[ielem,0]) for n in range(self.params['pecd_lmax']+1)) +\
                                            "\n")

            
            assert (self.params['pecd_lmax'] <= self.params['Leg_lmax'])
            
            for ielem, k in enumerate(k_pecd):
                print(  str('{:8.2f}'.format(k)) +\
                        " ".join('{:12.8f}'.format(bcoeff_R[ielem,n]/bcoeff_R[ielem,0]) for n in range(self.params['pecd_lmax']+1)) +\
                        " ".join('{:12.8f}'.format(bcoeff_L[ielem,n]/bcoeff_L[ielem,0]) for n in range(self.params['pecd_lmax']+1)) +\
                        "\n")

                pecd_list.append([   t,k, [bcoeff_R[ielem,n]/bcoeff_R[ielem,0] for n in range(self.params['pecd_lmax']+1)],
                                    [bcoeff_L[ielem,n]/bcoeff_L[ielem,0] for n in range(self.params['pecd_lmax']+1)],
                                    2.0 * bcoeff_R[ielem,1]/bcoeff_R[ielem,0] * 100.0])


        return pecd_list

    def PECD_plot2D(self,funcpars,polargrid,W):
        """ Produces contour plot for 2D spatial electron density f = rho(r,theta) """

        plot_params = funcpars['plot2D'][1] #all plot params

        """
        Args:
            polargrid: np.array of size (nptsr,nptsth,2): (r,theta) coordinates on a meshgrid
            rho: array of size (nptsr,nptsth): function values at each point of the polar meshgrid        Comments:
            plot_params: parameters of the plot loaded from GRAPHICS.py
        """

        figsizex    = plot_params['figsize_x'] #size of the figure on screen
        figsizey    = plot_params['figsize_y']  #size of the figure on screen
        resolution  = plot_params['resolution']  #resolution in dpi

        fig         = plt.figure(   figsize=(figsizex, figsizey), 
                                    dpi=resolution,
                                    constrained_layout=True)
                                    
        grid_fig    = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

        ax1         = fig.add_subplot(grid_fig[0, 0], projection='polar')

        cmap = matplotlib.cm.jet #jet, cool, etc
        norm = matplotlib.colors.Normalize(vmin=plot_params['vmin'], vmax=W.max())


        plot_W2D  = ax1.contourf(   polargrid[1], 
                                    polargrid[0], 
                                    W,  
                                    plot_params['ncont'], 
                                    cmap = 'jet', 
                                    vmin = plot_params['vmin'],
                                    vmax = W.max())
        

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


        ax1.set_ylim(polargrid[0].min(),funcpars['kmax']) #radial scale
        ax1.set_thetamin(polargrid[1].min()*180.0/np.pi)
        ax1.set_thetamax(polargrid[1].max()*180.0/np.pi)


        ax1.yaxis.grid(linewidth=0.5, alpha=0.5, color = '0.8', visible=True)
        ax1.xaxis.grid(linewidth=0.5, alpha=0.5, color = '0.8', visible=True)

        ax1.text(   0.0, 0.0, str('{:.1f}'.format(funcpars['t']/time_to_au) ) + " as", 
                    color = plot_params['time_colour'], fontsize = plot_params['time_size'],
                    alpha = 0.5,
                    transform = ax1.transAxes)

        
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

            fig.savefig(    fname       =   self.params['job_directory']  + "/graphics/momentum/"+
                                            plot_params['save_name'] + "_" + str(irun) + "_" +
                                            str('{:.1f}'.format(funcpars['t']/time_to_au) ) +
                                            "_" +
                                            self.params['helicity'] +
                                            ".pdf",
                                            
                            dpi         =   plot_params['save_dpi'],
                            orientation =   plot_params['save_orientation'],
                            bbox_inches =   plot_params['save_bbox_inches'],
                            pad_inches  =   plot_params['save_pad_inches']
                            )
        if funcpars['show'] == True:
            plt.show()
        
        plt.close()


    def W2Dav(self,funcpars):
        
        irun                    = self.params['irun']
        helicity                = self.pull_helicity()
        print("helicity = " + str(helicity))
        self.params['helicity'] = helicity

        Flm, kgrid = analysis_obj.calc_Flm(helicity)

        print("Calculating 2D electron momentum probability density phi-averaged")

        """ set up 1D momentum grids """

        if funcpars['k_grid']['type'] == "manual":
            # ktuple determines the range for which we calculate W2D. It also determines maximum plotting range.
            ktuple  = (funcpars['k_grid']['kmin'], funcpars['k_grid']['kmax'], funcpars['k_grid']['npts'])
            funcpars['ktulpe'] = ktuple
            kgrid1D         = np.linspace(ktuple[0], ktuple[1], ktuple[2], endpoint=True, dtype=float)

        elif funcpars['k_grid']['type'] == "automatic":
            # automatic radial momentum grid as given by the resolution of the FT 
            kgrid1D = kgrid
            ktuple  = (funcpars['k_grid']['kmin'], funcpars['k_grid']['kmax'], funcpars['k_grid']['npts'])
            funcpars['ktulpe'] = ktuple

        thtuple             = funcpars['th_grid']
        funcpars['thtuple'] = thtuple
        unity_vec           = np.linspace(0.0, 1.0, thtuple[2], endpoint=True, dtype=float)
        thgrid1D            = thtuple[1] * unity_vec
        

        """ generate 2D meshgrid for storing W2D """
        polargrid = np.meshgrid(kgrid1D, thgrid1D, indexing='ij')

        #print(polargrid[0].shape,polargrid[1].shape)
        #exit()

        obs_dict = {}
        obs_list = []
        """ set up time grids for evaluating wfn """
        tgrid_plot, plot_index   = self.setup_timegrids(self.params['momentum_analyze_times'])
        
        for i, (itime, t) in enumerate(zip(plot_index,list(tgrid_plot))):
  
            print(  "Generating W2D at time = " + str('{:6.2f}'.format(t/time_to_au) ) +\
                " " + str( self.params['time_units']) + " ----- " +\
                "time index = " + str(itime) )

            funcpars['t'] = t

            Flm_t = Flm[i*(self.params['bound_lmax']+1)**2:(i+1)*(self.params['bound_lmax']+1)**2 ]

            Wav = self.W2Dav_calc( funcpars,
                                    Flm_t,
                                    polargrid)
        
            obs_dict['Wav'] = Wav
            if funcpars['plot'][0] == True:

                self.W2Dav_plot(funcpars,polargrid,Wav)

            if funcpars['save'] == True:
   
                with open( self.params['job_directory'] +  "W2Dav" + "_" + str(irun) + "_" + str('{:.1f}'.format(t/time_to_au) ) +\
                    "_" + helicity + ".dat" , 'w') as rhofile:   
                    np.savetxt(rhofile, Wav, fmt = '%10.4e')

            if funcpars['legendre'] == True:
                # perform legendre expansion of W2Dav
                bcoeff, kgrid = self.legendre_expansion(funcpars,polargrid,{'av':Wav})
                obs_dict['bcoeff']  = bcoeff
                obs_dict['kgrid']   = kgrid

            if funcpars['PES']  == True:
                PES_av = self.PESav(funcpars,polargrid,Wav)
                obs_dict['PES_av'] = PES_av

            obs_list.append([i,t,obs_dict])

        return obs_list, polargrid


    def W2Dav_calc(self, funcpar, Flm, polargrid):
        """calculate numerically W2D for a sequence of phi angles and return averaged W2Dav"""

        print("Evaluation planes for W2D: Z")
        Wav = np.zeros((polargrid[0].shape[0],polargrid[1].shape[1]), dtype=float)

        phimin = 0.0
        phimax = 2.0 * np.pi
        phigrid = np.linspace(phimin, phimax, funcpar['nphi_pts'], endpoint=False)
        
        for iphi in range(phigrid.shape[0]):
            print('iphi = ' + str(iphi) + ', phi = ' + str('{:.2}'.format(phigrid[iphi])))
            Ymat            = self.calc_spharm_array(self.params['bound_lmax'], 'Z', polargrid, phigrid[iphi])
            W2D             = self.calc_FT_3D_hankel(Flm, Ymat)
            Wav             += np.abs(W2D)**2/np.max(np.abs(W2D)**2)

        return Wav / funcpar['nphi_pts']



    def W2Dav_plot(self,funcpars,polargrid,W2D): 
        """ Produces contour plot for 2D spatial electron density f = rho(r,theta) """

        plot_params = funcpars['plot'][1] #all plot params
        ktuple      = funcpars['ktulpe'] #range for k
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

        fig         = plt.figure(   figsize=(figsizex, figsizey), 
                                    dpi=resolution,
                                    constrained_layout=True)
                                    
        grid_fig    = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

        ax1         = fig.add_subplot(grid_fig[0, 0], projection='polar')

        cmap = matplotlib.cm.jet #jet, cool, etc
        norm = matplotlib.colors.Normalize(vmin=plot_params['vmin'], vmax=plot_params['vmax'])


        ax1.set_ylim(ktuple[0],ktuple[1]) #radial scale
        ax1.set_thetamin(thtuple[0]*180.0/np.pi)
        ax1.set_thetamax(thtuple[1]*180.0/np.pi)


        plot_params['thticks']  = list(np.linspace(thtuple[0],thtuple[1],plot_params['nticks_th']))
        plot_params['rticks']   = list(np.linspace(ktuple[0],ktuple[1],plot_params['nticks_rad'])) 
                            

        plot_W2D  = ax1.contourf(   polargrid[1], 
                                    polargrid[0], 
                                    W2D,  
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



        ax1.yaxis.grid(linewidth=0.5, alpha=0.5, color = '0.8', visible=True)
        ax1.xaxis.grid(linewidth=0.5, alpha=0.5, color = '0.8', visible=True)

        ax1.text(   0.0, 0.0, str('{:.1f}'.format(funcpars['t']/time_to_au) ) + " as", 
                    color = plot_params['time_colour'], fontsize = plot_params['time_size'],
                    alpha = 0.5,
                    transform = ax1.transAxes)

        
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

            fig.savefig(    fname       =   self.params['job_directory']  + "/graphics/momentum/"+
                                            plot_params['save_name'] + "_" + str(irun) + "_" +
                                            str('{:.1f}'.format(funcpars['t']/time_to_au) ) +
                                            "_" +
                                            self.params['helicity'] +
                                            ".pdf",
                                            
                            dpi         =   plot_params['save_dpi'],
                            orientation =   plot_params['save_orientation'],
                            bbox_inches =   plot_params['save_bbox_inches'],
                            pad_inches  =   plot_params['save_pad_inches']
                            )
        if funcpars['show'] == True:
            plt.show()
        
        plt.close()





    def W2D(self,funcpars):

        Flm     = self.params['Flm']
        kgrid   = self.params['momentumgrid']

        print("Calculating 2D electron momentum probability density")
        
        irun = self.params['irun']
        helicity  = self.pull_helicity()
        self.params['helicity'] = helicity
        print("helicity = " + str(helicity))
        #exit()
        """ set up 1D momentum grids """

        if funcpars['k_grid']['type'] == "manual":
            # ktuple determines the range for which we calculate W2D. It also determines maximum plotting range.
            ktuple  = (funcpars['k_grid']['kmin'], funcpars['k_grid']['kmax'], funcpars['k_grid']['npts'])
            funcpars['ktulpe'] = ktuple
            kgrid1D         = np.linspace(ktuple[0], ktuple[1], ktuple[2], endpoint=True, dtype=float)

        elif funcpars['k_grid']['type'] == "automatic":
            # automatic radial momentum grid as given by the resolution of the FT 
            kgrid1D = kgrid
            ktuple  = (funcpars['k_grid']['kmin'], funcpars['k_grid']['kmax'], funcpars['k_grid']['npts'])
            funcpars['ktulpe'] = ktuple

        thtuple             = funcpars['th_grid']
        funcpars['thtuple'] = thtuple
        unity_vec           = np.linspace(0.0, 1.0, thtuple[2], endpoint=True, dtype=float)
        thgrid1D            = thtuple[1] * unity_vec
        

        """ generate 2D meshgrid for storing W2D """
        polargrid = np.meshgrid(kgrid1D, thgrid1D, indexing='ij')

        #print(polargrid[0].shape,polargrid[1].shape)
        #exit()

        
        """ set up time grids for evaluating wfn """
        tgrid_plot, plot_index   = self.setup_timegrids(self.params['momentum_analyze_times'])
        
        for i, (itime, t) in enumerate(zip(plot_index,list(tgrid_plot))):
  
            print(  "Generating W2D at time = " + str('{:6.2f}'.format(t/time_to_au) ) +\
                " " + str( self.params['time_units']) + " ----- " +\
                "time index = " + str(itime) )

            funcpars['t'] = t

            Flm_t = Flm[i*(self.params['bound_lmax']+1)**2:(i+1)*(self.params['bound_lmax']+1)**2 ]

            W2Ddir = self.W2D_calc( funcpars,
                                    Flm_t,
                                    polargrid)


            if funcpars['plot'][0] == True:

                for elem in W2Ddir.items():

                    plane       = elem[0]
                    W           = elem[1]
                    print(W.shape)

                    funcpars['plane_split'] = self.split_word(plane)

                    # call plotting function
                    self.W2D_plot(funcpars,polargrid,W)


            if funcpars['save'] == True:
                for elem in W2Ddir.items():
                    with open( self.params['job_directory'] +  "W2D" + str(irun)  + "_" + str('{:.1f}'.format(t/time_to_au) ) +\
                        "_" + helicity + ".dat" , 'w') as rhofile:   
                        np.savetxt(rhofile, elem[1], fmt = '%10.4e')

            if funcpars['legendre'] == True:
                # perform legendre expansion of W2D
                self.legendre_expansion(funcpars,polargrid,W2Ddir)

            if funcpars['PES']  == True:
                self.PES(funcpars,polargrid,W2Ddir)

        
    def W2D_calc(self, funcpar, Flm, polargrid):
        """calculate numerically W2D for fixed phi angle"""

        plane       = funcpar['plane']

        W2Ddir = {} # dictionary containing numpy arrays representing the electron's momentum density on (r,theta) meshgrid

        for elem in plane:

            print("Evaluation plane for W2D: " + elem)

            Ymat            = self.calc_spharm_array(self.params['bound_lmax'], elem, polargrid)
            W2D             = self.calc_FT_3D_hankel(Flm, Ymat)
            W2Ddir[elem]    = np.abs(W2D)**2/np.max(np.abs(W2D)**2)

        return W2Ddir



    def W2D_plot(self,funcpars,polargrid,W2D): 
        """ Produces contour plot for 2D spatial electron density f = rho(r,theta) """

        plot_params = funcpars['plot'][1] #all plot params
        ktuple      = funcpars['ktulpe'] #range for k
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

        fig         = plt.figure(   figsize=(figsizex, figsizey), 
                                    dpi=resolution,
                                    constrained_layout=True)
                                    
        grid_fig    = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

        ax1         = fig.add_subplot(grid_fig[0, 0], projection='polar')

        cmap = matplotlib.cm.jet #jet, cool, etc
        norm = matplotlib.colors.Normalize(vmin=plot_params['vmin'], vmax=plot_params['vmax'])


        ax1.set_ylim(ktuple[0],ktuple[1]) #radial scale
        ax1.set_thetamin(thtuple[0]*180.0/np.pi)
        ax1.set_thetamax(thtuple[1]*180.0/np.pi)


        plot_params['thticks']  = list(np.linspace(thtuple[0],thtuple[1],plot_params['nticks_th']))
        plot_params['rticks']   = list(np.linspace(ktuple[0],ktuple[1],plot_params['nticks_rad'])) 
                            

        plot_W2D  = ax1.contourf(   polargrid[1], 
                                    polargrid[0], 
                                    W2D,  
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
        ax1.xaxis.grid(linewidth=0.5, alpha=0.5, color = '0.8', visible=True)

        ax1.text(   0.0, 0.0, str('{:.1f}'.format(funcpars['t']/time_to_au) ) + " as", 
                    color = plot_params['time_colour'], fontsize = plot_params['time_size'],
                    alpha = 0.5,
                    transform = ax1.transAxes)

        
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

            fig.savefig(    fname       =   params['job_directory']  + "/graphics/momentum/"+
                                            plot_params['save_name'] + "_" +
                                            funcpars['plane_split'][1]+
                                            funcpars['plane_split'][0]+ "_" + str(irun) + "_" +
                                            str('{:.1f}'.format(funcpars['t']/time_to_au) ) +
                                            "_" +
                                            params['helicity'] +
                                            ".pdf",
                                            
                            dpi         =   plot_params['save_dpi'],
                            orientation =   plot_params['save_orientation'],
                            bbox_inches =   plot_params['save_bbox_inches'],
                            pad_inches  =   plot_params['save_pad_inches']
                            )
        if funcpars['show'] == True:
            plt.show()
        
        plt.close()



    def PES(self,funcpars,grid,fdir):

        for elem in fdir.items():

            f       = elem[1]
            plane   = elem[0]

            kgrid       = grid[0]
            thetagrid   = grid[1]

            Lmax    = self.params['pes_lmax'] 
            nleg     = Lmax
            print(nleg)
            x, w    = np.polynomial.legendre.leggauss(nleg)
            w       = w.reshape(nleg,-1)

            """ Interpolate f(k,theta)"""
            W_interp    = interpolate.RectBivariateSpline(kgrid[:,0], thetagrid[0,:], f[:,:], kx=3, ky=3)


            print("*** calculating photo-electron spectrum ***")
            nkpoints    = params['pes_npts'] 
            pes_kgrid   = np.linspace(0.05, params['pes_max_k'], nkpoints)
            spectrum    = np.zeros(nkpoints, dtype = float)
            
            
            for ipoint,k in enumerate(list(pes_kgrid)):   
            
                W_interp1        = W_interp(k,-np.arccos(x)).reshape(nleg,-1) 
                spectrum[ipoint] = np.sum(w[:,0] * W_interp1[:,0] * np.sin(np.arccos(x)) ) 
            
            self.PES_plot(funcpars,spectrum,pes_kgrid)


    def PESav(self,funcpars,grid,Wav):
        """ photo-electron spectrum from phi-averaged momentum distribution"""
    
        kgrid       = grid[0]
        thetagrid   = grid[1]

        Lmax        = self.params['pes_lmax'] 
        nleg        = Lmax

        x, w        = np.polynomial.legendre.leggauss(nleg)
        w           = w.reshape(nleg,-1)

        """ Interpolate Wav(k,theta)"""
        W_interp    = interpolate.RectBivariateSpline(kgrid[:,0], thetagrid[0,:], Wav[:,:], kx=3, ky=3)


        print("*** calculating photo-electron spectrum ***")
        nkpoints    = params['pes_npts'] 
        pes_kgrid   = np.linspace(0.05, params['pes_max_k'], nkpoints)
        spectrum    = np.zeros(nkpoints, dtype = float)
        
        
        for ipoint,k in enumerate(list(pes_kgrid)):   
        
            W_interp1        = W_interp(k,-np.arccos(x)).reshape(nleg,-1) 
            spectrum[ipoint] = np.sum(w[:,0] * W_interp1[:,0] * np.sin(np.arccos(x)) ) 
        
        self.PES_plot(funcpars,spectrum,pes_kgrid)


    def PES_plot(self,funcpars,spectrum,kgrid):

        """ Produces plot of the PES """
       
        plot_params = funcpars['PES_params']['plot'][1] #all plot params

        """
        Args:
            kgrid: np.array of size (nkpoints): momentum grid in a.u.
            spectrum: array of size (nkpoints): 1D PES
            plot_params: parameters of the plot loaded from GRAPHICS.py
        """

        figsizex    = plot_params['figsize_x']  # size of the figure on screen
        figsizey    = plot_params['figsize_y']  # size of the figure on screen
        resolution  = plot_params['resolution'] # resolution in dpi

        fig         = plt.figure(figsize=(figsizex, figsizey), dpi=resolution,
                        constrained_layout=True)
        grid_fig    = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

        ax1         = fig.add_subplot(grid_fig[0, 0], projection='rectilinear')

        if funcpars['PES_params']['k-axis'] == 'momentum':
            grid_plot = kgrid
        elif funcpars['PES_params']['k-axis'] == 'energy':
            grid_plot = (0.5*kgrid**2)*CONSTANTS.au_to_ev
        else:
            raise ValueError("incorrect k-axis specification")

        if funcpars['PES_params']['y-axis'] == 'unit':
            func_plot = kgrid * spectrum

        elif funcpars['PES_params']['y-axis'] == 'log':
            func_plot =   kgrid * spectrum
            ax1.set_yscale('log')
        else:
            raise ValueError("incorrect y-axis specification")

        if funcpars['PES_params']['normalize'] == True:
            func_plot /= func_plot.max()
    
        plot_PES    = ax1.plot( grid_plot,
                                func_plot, 
                                label = plot_params['plot_label'],
                                marker = plot_params['plot_marker'], 
                                color = plot_params['plot_colour'] )

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


        if funcpars['PES_params']['k-axis'] == 'momentum':
            plt.xlabel("momentum (a.u)")
        elif funcpars['PES_params']['k-axis'] == 'energy':
            plt.xlabel("energy (eV)")

        if funcpars['PES_params']['y-axis'] == 'unit':
            if funcpars['PES_params']['normalize'] == True:
                plt.ylabel("cross section " +r"$\sigma(k)$" +  " (normalized)")
            else:
                plt.ylabel("cross section " +r"$\sigma(k)$" )

        elif funcpars['PES_params']['y-axis'] == 'log':
            if funcpars['PES_params']['normalize'] == True:
                plt.ylabel("cross section "+r"$log(\sigma(k))$" + " (normalized)")
            else:
                plt.ylabel("cross section "+r"$log(\sigma(k))$")


        ax1.yaxis.grid(linewidth=0.5, alpha=0.5, color = '0.8', visible=True)
        ax1.xaxis.grid(linewidth=0.5, alpha=0.5, color = '0.8', visible=True)


        ax1.text(   0.0, 0.0, str('{:.1f}'.format(funcpars['t']/time_to_au) ) + " as", 
                    color = plot_params['time_colour'], fontsize = plot_params['time_size'],
                    alpha = 0.5,
                    transform = ax1.transAxes)

        #custom ticks and labels:
        #ax1.set_xticks(plot_params['thticks']) #positions of th-ticks
        #ax1.set_yticks(plot_params['rticks']) #positions of r-ticks

        #ax1.set_xticklabels(plot_params['thticks'],fontsize=8) #th-ticks labels
        #ax1.set_yticklabels(plot_params['rticks'],fontsize=8) #r-ticks labels

        #ax1.xaxis.set_major_formatter(FormatStrFormatter(plot_params['xlabel_format'])) #set tick label formatter 
        #ax1.yaxis.set_major_formatter(FormatStrFormatter(plot_params['ylabel_format']))

        if plot_params['save'] == True:
            if funcpars['name'] == 'W2D':
                fig.savefig(    fname       =   params['job_directory']  + "/graphics/momentum/"+
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
            elif  funcpars['name'] == 'W2Dav':
                fig.savefig(    fname       =   params['job_directory']  + "/graphics/momentum/"+
                                            plot_params['save_name'] + "_" +
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
        plt.legend()  
        plt.close()


    def calc_fftcart_psi_3d(self,params, maparray, Gr, psi, chilist):
        coeff_thr = 1e-3
        ncontours = 20

        nlobs   = params['bound_nlobs']
        nbins   = params['prop_nbins'] 
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




def plot_W_3D_num(self,params, maparray_chi, maparray_global, psi, chilist, phi0 = 0.0):
    ncontours = 100
    grid_theta, grid_r = calc_grid_for_FT(params)

    #calculate partial waves on radial grid
    Plm = calc_partial_waves(chilist, grid_r, params['bound_lmax'], psi, maparray_global, maparray_chi,  ipoint_cutoff)

    #calculate Hankel transforms on appropriate k-vector grid
    Flm, kgrid = calc_hankel_transforms(Plm, grid_r)

    FT, kgrid = calc_FT_3D_hankel(Plm, Flm, kgrid, params['bound_lmax'], grid_theta, grid_r, maparray_chi, maparray_global, psi, chilist, phi0 )

    fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    axft = fig.add_subplot(spec[0, 0], projection='polar')

    Nrad = len(kgrid)
    Nang = Nrad

    kmesh, thetamesh = np.meshgrid(kgrid,grid_theta)

    axft.set_ylim(0,1) #radial extent
    line_ft = axft.contourf(thetamesh, kmesh, np.abs(FT)**2/np.max(np.abs(FT)**2), 
                            ncontours, cmap = 'jet') #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    plt.colorbar(line_ft, ax=axft, aspect=30)
    
    plt.legend()   
    plt.show()  





def calc_W_3D_analytic(self,lmax, grid_theta, grid_r, maparray_chi, maparray_global, psi, chilist, phi0 = 0.0 ):
    """ returns: square modulus of fourier transform inside a ball grid (r,theta,phi) calculated with an analytic expression"""
    npts      = grid_r.size
    #Precompute spherical harmonics on the theta grid
    SP = precompute_spharm(grid_theta,lmax, phi0)

    #plt.plot(grid_theta,PLOTS.spharm(2, 1, grid_theta, phi0))
    #plt.show()

    #calculate partial waves on radial grid
    Plm = calc_partial_waves(chilist, grid_r, lmax, psi, maparray_global, maparray_chi,  ipoint_cutoff)

    #calculate Hankel transforms on appropriate k-vector grid
    Flm, kgrid = calc_hankel_transforms(Plm, grid_r)

    W = np.zeros((npts, npts), dtype = complex)

    #for i in range(npts):
    #    print(i)
    for elem1 in Flm:
        print(elem1[0],elem1[1])
        for elem2 in Flm:
            for L in range( int(np.abs(elem2[0]-elem1[0])), elem1[0] + elem2[0] + 1 ):
                for M in range(-L,L+1):
                    for i in range(npts):
                        #if abs(elem2[1]-elem1[1]) <= L:
                        W[i,:] +=  (-1.0)**(elem2[0]+elem1[1]) * ((1j)**(elem1[0] + elem2[0])) * \
                                    np.conj(elem1[2][:]) * elem2[2][:] * np.sqrt( (2*elem1[0]+1) * (2*elem2[0]+1) / (2*L+1) ) * \
                                    float(N(clebsch_gordan(elem1[0], elem2[0], L, 0, 0, 0))) * float(N(clebsch_gordan(elem1[0], elem2[0], L, -elem1[1], elem2[1], M))) *\
                                    PLOTS.spharm(L, M, grid_theta[i] , phi0)
                                #SP[ str(L) +',' + str(elem1[1]-elem2[1]) ][:]
                                    #float(N(CG(elem1[0],0,elem2[0],0,L,0).doit())) * float(N(CG(elem1[0],-elem1[1],elem2[0],elem2[1],L,M).doit())) *\
                                    #
                                    # 
    print(W)
    return W, kgrid


def plot_W_3D_analytic(self,params, maparray_chi, maparray_global, psi, chilist, phi0 = 0.0):
    ncontours = 100
    grid_theta, grid_r = calc_grid_for_FT(params)

    W, kgrid = calc_W_3D_analytic(params['bound_lmax'], grid_theta, grid_r, maparray_chi, maparray_global, psi, chilist, phi0 )

    fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    axft = fig.add_subplot(spec[0, 0], projection='polar')

    Nrad = len(kgrid)
    Nang = Nrad

    kmesh, thetamesh = np.meshgrid(kgrid,grid_theta)

    axft.set_ylim(0,1) #radial extent
    line_ft = axft.contourf(thetamesh, kmesh, W/np.max(W), 
                            ncontours, cmap = 'jet') #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    plt.colorbar(line_ft, ax=axft, aspect=30)
    
    plt.legend()   
    plt.show()  






def calc_W_av_phi_analytic(self,params, maparray_chi, maparray_global, psi, chilist):
    """ returns: square modulus of fourier transform inside a disk (r,theta) calculated with analytic expression
        and averaged over phi"""

    grid_theta, grid_r = calc_grid_for_FT(params)
    npts      = grid_r.size

    #calculate partial waves on radial grid
    Plm = calc_partial_waves(chilist, grid_r, params['bound_lmax'], psi, maparray_global, maparray_chi,  ipoint_cutoff)

    #calculate Hankel transforms on appropriate k-vector grid
    Flm, kgrid = calc_hankel_transforms(Plm, grid_r)

    Wav = np.zeros((npts , npts), dtype = complex)


    for elem1 in Flm:
        print(elem1[0],elem1[1])
        for elem2 in Flm:
            if elem1[1]==elem2[1]:
                for L in range(np.abs(elem1[0]-elem2[0]),elem1[0]+elem2[0]+1):
                    for i in range(npts):
                        Wav[i,:] +=  (-1.0)**(elem2[0]+elem1[1]) * ((1j)**(elem1[0]+elem2[0])) * \
                                    np.conj(elem1[2][:]) * elem2[2][:] *\
                                    float(N(clebsch_gordan(elem1[0], elem2[0], L, 0, 0, 0))) * \
                                    float(N(clebsch_gordan(elem1[0], elem2[0], L, -elem1[1], elem2[1], 0))) *\
                                    eval_legendre(L, np.cos(grid_theta[i]))

    print(Wav)
    return Wav, kgrid

def plot_W_2D_av_phi_analytic(self,params, maparray_chi, maparray_global, psi, chilist):
    ncontours = 100
    grid_theta, grid_r = calc_grid_for_FT(params)

    Wav, kgrid  = calc_W_av_phi_analytic(params, maparray_chi, maparray_global, psi, chilist)

    fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    axft = fig.add_subplot(spec[0, 0], projection='polar')

    Nrad = len(kgrid)

    kmesh, thetamesh = np.meshgrid(kgrid,grid_theta)

    axft.set_ylim(0,1) #radial extent
    line_ft = axft.contourf(thetamesh, kmesh, Wav/np.max(Wav), 
                            ncontours, cmap = 'jet') #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    plt.colorbar(line_ft, ax=axft, aspect=30)
    
    plt.legend()   
    plt.show()  


def calc_ftpsi_2d(self,params, maparray, Gr, psi, chilist):
   
    coeff_thr = 1e-5
    ncontours = 100

    nlobs   = params['bound_nlobs']
    nbins   = params['bound_nbins'] 
    npoints = 200
    rmax    = nbins * params['bound_binw']
    rmin    = 25.0

    fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    axft = fig.add_subplot(spec[0, 0])

    cart_grid = np.linspace(rmin, rmax*0.95, npoints, endpoint=True, dtype=float)

    y2d, z2d = np.meshgrid(cart_grid,cart_grid)

    thetagrid = np.arctan2(y2d,z2d)
    rgrid = np.sqrt(y2d**2 + z2d**2)

    x   =  np.zeros(nlobs)
    w   =  np.zeros(nlobs)
    x,w =  GRID.gauss_lobatto(nlobs,14)
    w   =  np.array(w)
    x   =  np.array(x) # convert back to np arrays

    y = np.zeros((len(y2d),len(z2d)), dtype=complex)

    phi0 = 0.0 * np.pi/2 #fixed polar angle
    
    for ielem, elem in enumerate(maparray):
        if np.abs(psi[2*ielem]+1j*psi[2*ielem + 1]) > coeff_thr:
            print(str(elem) + str(psi[ielem]))

            #chir = flist[elem[2]-1](rgrid) #labelled by xi

            for i in range(len(cart_grid)):
     
                y[i,:] +=  ( psi[2*ielem] + 1j * psi[2*ielem + 1] ) * PLOTS.spharm(elem[3], elem[4], thetagrid[i,:], phi0) * \
                        chilist[elem[2]-1](rgrid[i,:]) #chi(elem[0], elem[1], rang[:], Gr, w, nlobs, nbins) 


    fty = fftn(y)
    print(fty)

    ft_grid = np.linspace(-1.0/(rmax), 1.0/(rmax), npoints, endpoint=True, dtype=float)

    yftgrid, zftgrid = np.meshgrid(ft_grid,ft_grid)

    line_ft = axft.contourf(yftgrid, zftgrid , fty[:npoints].imag/np.max(np.abs(fty)), 
                                        ncontours, cmap = 'jet', vmin=-0.2, vmax=0.2) #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    plt.colorbar(line_ft, ax=axft, aspect=30)
    
    #axradang_r.set_yticklabels(list(str(np.linspace(rmin,rmax,5.0)))) # set radial tick label
    plt.legend()   
    plt.show()  

    return fty, yftgrid, zftgrid 






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
    params['maparray_global'], params['Nbas_global']    = MAPPING.GENMAP_FEMLIST(   params['FEMLIST_PROP'],
                                                                                    params['bound_lmax'],
                                                                                    params['map_type'],
                                                                                    params['job_directory'] )

    params['maparray_chi'], params['Nbas_chi']          = MAPPING.GENMAP_FEMLIST(   params['FEMLIST_PROP'],  
                                                                                    0,
                                                                                    params['map_type'], 
                                                                                    path )

    params['Gr'], params['Nr ']                         = GRID.r_grid(              params['bound_nlobs'], 
                                                                                    params['prop_nbins'] , 
                                                                                    params['bound_binw'],  
                                                                                    params['bound_rshift'] )

    params['Gr_prim'], params['Nr_prim']                = GRID.r_grid_prim(         params['bound_nlobs'], 
                                                                                    params['prop_nbins'], 
                                                                                    params['bound_binw'], 
                                                                                    params['bound_rshift'] )

    params['chilist']                                   = PLOTS.interpolate_chi(    params['Gr_prim'], 
                                                                                    params['bound_nlobs'], 
                                                                                    params['prop_nbins'], 
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
        
        # Calculate an array of Hankel transforms on the momentum grid (1D, 2D or 3D) for all selected times
        
        
        momentumobs     = momentumfuncs(params)

        for elem in params['analyze_momentum']:

            func = getattr(momentumobs,elem['name'])
            print("Calling momentum function: " + str(elem['name']))
            func(elem)
        


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
    """ Consolidate quanitites averaged over orientations """
    obj     = analysis(params)
    ibcoeff = 9
    obj.barray_plot_2D(grid_euler,ibcoeff,params['bcoeffs'])
    #exit()
