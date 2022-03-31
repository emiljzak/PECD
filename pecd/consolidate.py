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
import rotdens
import constants


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
    
        #2. setup time grid
        self.tgrid               = self.calc_tgrid()
        #3. setup time grid indices to analyze - they correspond to the respective analyze times
        self.tgrid_plot_index    =  self.calc_plot_times(self.tgrid,params['momentum_analyze_times']) 
        self.tgrid_plot_time     = self.tgrid[self.tgrid_plot_index]
        print("times for which momentum functions are analyzed: " + str(self.tgrid_plot_time/time_to_au))
        

            """ set up time grids for evaluating rho1D from the propagated wavefunction """
        tgrid,dt = self.calc_tgrid()

        plot_index = self.calc_plot_times(tgrid,dt,self.params['space_analyze_times']) #plot time indices

        tgrid_plot = tgrid[plot_index]

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


    def calc_pecd2D_av(self,N_Euler,grid_euler,rho):
        """ Polar plot (k,theta) of 3D orientation averaged PECD
        
            The function reads W2Dav distributions for 'L' and 'R' helicity and for each orientation and sums the respective distributions.
            Finally orientation averaged W2D('R') is substracted from W2D('L')
        
        """

        #generate polargrid

        #read W2Dav files

        #set up time at which we calculate W
        index_time = self.params['index_time'][0]
        t = timegrid[index_time]


        for helicity in self.params['helicity_consolidate']:
            for ipoint in range(N_Euler):
                #read W2D(helicity,ipoint)
        
                with open( self.params['job_directory'] +  "W2Dav" + "_" + str(ipoint) + "_" + str('{:.1f}'.format(t/time_to_au) ) + "_" + helicity + ".dat" , 'r') as Wavfile:   
                    W2D = np.loadtxt(Wavfile)
                print(W2D)
                print(W2D.shape)

                exit()
                W += rho[ipoint] * W2D


        self.pecd_plot2D(self.params['PECD_av'],polargrid,W)

        return W


    def pecd_plot2D(self,funcpars,ft_polargrid,W): 
        """Produces a contour 2D plot for 3D averaged electron's momentum distribution 
        
        .. warning:: In the present implementation the 2D controur plot is produced for the **original FT** grid, not for the grid defined by the user. User-defined grid determines ranges and ticks only. 

        .. note:: `contourf` already performs interpolation of W2D. There is no need for evaluation of W2D on a finer grid than ft_polargrid.
        """
        #radial grid
        if funcpars['k_grid']['type'] == "manual":
            
            # ktuple determines the range for which we PLOT W2D. ft_grid is the grid over which W2Dav is evaluated and kept in memory.
            ktuple  = (funcpars['k_grid']['kmin'], funcpars['k_grid']['kmax'])

        elif funcpars['k_grid']['type'] == "automatic":

            # automatic radial momentum grid as given by the resolution of the FT 
            ktuple  = (ft_polargrid[0].min(), ft_polargrid[0].max())

        thtuple             = (funcpars['th_grid']['thmin'], funcpars['th_grid']['thmax'], funcpars['th_grid']['FT_npts_th'])

        plot_params = funcpars['plot'][1] #all plot params

        figsizex    = plot_params['figsize_x'] #size of the figure on screen
        figsizey    = plot_params['figsize_y']  #size of the figure on screen
        resolution  = plot_params['resolution']  #resolution in dpi

        fig         = plt.figure(   figsize = (figsizex, figsizey), 
                                    dpi = resolution,
                                    constrained_layout = True)
                                    
        grid_fig    = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
        ax1         = fig.add_subplot(grid_fig[0, 0], projection='polar')

        cmap = matplotlib.cm.jet #jet, cool, etc
        norm = matplotlib.colors.Normalize(vmin=plot_params['vmin'], vmax=plot_params['vmax'])


        ax1.set_ylim(ktuple[0],ktuple[1]) #radial scale
        ax1.set_thetamin(thtuple[0]*180.0/np.pi)
        ax1.set_thetamax(thtuple[1]*180.0/np.pi)


        plot_params['thticks']  = list(np.linspace(thtuple[0],thtuple[1],plot_params['nticks_th']))
        plot_params['rticks']   = list(np.linspace(ktuple[0],ktuple[1],plot_params['nticks_rad'])) 

      

        plot_W2D  = ax1.contourf(   ft_polargrid[1], 
                                    ft_polargrid[0], 
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
                                            self.helicity +
                                            ".pdf",
                                            
                            dpi         =   plot_params['save_dpi'],
                            orientation =   plot_params['save_orientation'],
                            bbox_inches =   plot_params['save_bbox_inches'],
                            pad_inches  =   plot_params['save_pad_inches']
                            )
        if funcpars['show'] == True:
            plt.show()
        
        plt.close()

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

        default_size_min = int(1*10**8)
        default_size_max = int(1.05*10**8)
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
            if ielem%self.params['res_jobs_per_batch'] == 0 and ielem>0:
                print(awkward_filesL[ielem-self.params['res_jobs_per_batch']:ielem])
            if ielem > len(awkward_filesL)-self.params['res_jobs_per_batch']:
                print(awkward_filesL[ielem:])
                break
        print("R: awkward files:")
        for ielem,elem in enumerate(awkward_filesR):
            if ielem%self.params['res_jobs_per_batch'] == 0 and ielem>0:
                print(awkward_filesR[ielem-self.params['res_jobs_per_batch']:ielem])                      
            if ielem > len(awkward_filesR)-self.params['res_jobs_per_batch']:
                print(awkward_filesR[ielem:])
                break

        print("L: missing files:")
        for ielem,elem in enumerate(missing_filesL):
            if ielem%self.params['res_jobs_per_batch'] == 0 and ielem>0:
                print(missing_filesL[ielem-self.params['res_jobs_per_batch']:ielem])
            if ielem > len(missing_filesL)-self.params['res_jobs_per_batch']:
                print(missing_filesL[ielem:])
                break

        print("R: missing files:")
        for ielem,elem in enumerate(missing_filesR):
            if ielem%self.params['res_jobs_per_batch'] == 0 and ielem>0:
                print(missing_filesR[ielem-self.params['res_jobs_per_batch']:ielem])
            if ielem > len(missing_filesR)-self.params['res_jobs_per_batch']:
                print(missing_filesR[ielem:])
                break
        

        #save lists in restart files
        with open('restart_R.dat', 'w') as restart_file:
            
            for ielem,elem in enumerate(missing_filesR):
                if ielem%self.params['res_jobs_per_batch'] == 0 and ielem>0:
                   restart_file.write(" ".join('{:4d}'.format(v) for v in missing_filesR[ielem-self.params['res_jobs_per_batch']:ielem])+"\n")
                if ielem > len(missing_filesR)-self.params['res_jobs_per_batch']:           
                    restart_file.write(" ".join('{:4d}'.format(v) for v in  missing_filesR[ielem:] )+"\n")
                    break

            for ielem,elem in enumerate(awkward_filesR):
                if ielem%self.params['res_jobs_per_batch'] == 0 and ielem>0:
                    restart_file.write(" ".join('{:4d}'.format(v) for v in  awkward_filesR[ielem-self.params['res_jobs_per_batch']:ielem] )+"\n")
                   
                if ielem > len(awkward_filesR)-self.params['res_jobs_per_batch']:                    
                    restart_file.write(" ".join('{:4d}'.format(v) for v in  awkward_filesR[ielem:])+"\n")
                    break


        with open('restart_L.dat', 'w') as restart_file:
            
            for ielem,elem in enumerate(missing_filesL):
                if ielem%self.params['res_jobs_per_batch'] == 0 and ielem>0:
                   restart_file.write(" ".join('{:4d}'.format(v) for v in missing_filesL[ielem-self.params['res_jobs_per_batch']:ielem])+"\n")
                if ielem > len(missing_filesL)-self.params['res_jobs_per_batch']:           
                    restart_file.write(" ".join('{:4d}'.format(v) for v in  missing_filesL[ielem:] )+"\n")
                    break

            for ielem,elem in enumerate(awkward_filesL):
                if ielem%self.params['res_jobs_per_batch'] == 0 and ielem>0:
                    restart_file.write(" ".join('{:4d}'.format(v) for v in  awkward_filesL[ielem-self.params['res_jobs_per_batch']:ielem] )+"\n")
                   
                if ielem > len(awkward_filesL)-self.params['res_jobs_per_batch']:                    
                    restart_file.write(" ".join('{:4d}'.format(v) for v in  awkward_filesL[ielem:])+"\n")
                    break


        exit()
        return missing_files,awkward_files




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

    time_to_au = constants.time_to_au[ params['time_units'] ]

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


    if params['density_averaging'] == True:
        print("\n")
        print("Reading grid of molecular orientations (Euler angles)...")
        print("\n")

        GridObjEuler = wavefunction.GridEuler(  params['N_euler'],
                                                params['N_batches'],
                                                params['orient_grid_type'])
        
        grid_euler, N_Euler, N_per_batch  = GridObjEuler.read_euler_grid()

        WDMATS  = GridObjEuler.gen_wigner_dmats( N_Euler,grid_euler, params['Jmax'])
        #calculate rotational density at grid (alpha, beta, gamma) = (n_grid_euler, 3)

        grid_rho, rho = rotdens.calc_rotdens(   grid_euler,
                                                WDMATS,
                                                params) 

        print("rho:")
        print(rho)
        #exit()

    Obs.plot_pecd_av(N_Euler,grid_euler,rho)
    exit()


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
    #polar plot of orientation averaged PECD (L-R distributions)

    #
    print("b1(R)/b_av= " + str(b_av_dict["R"][1]/b_av_dict["R"][0]))
    print("b1(L)/b_av= " + str(b_av_dict["L"][1]/b_av_dict["R"][0]))