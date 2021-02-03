#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
import sys
import matelem
import time
import quadpy

from field import Field
from basis import radbas
from matelem import mapping

import constants 

from scipy.linalg import expm
from scipy.special import genlaguerre,sph_harm,roots_genlaguerre,factorial,roots_hermite
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker

from numpy.polynomial.laguerre import laggauss

class propagate(radbas,mapping):

    def __init__(self):
        pass

    def prop_wf(self,params,psi0):
        """ main method for propagating the wavefunction"""
        """params (dict):      dictionary with all relevant numerical parameters of the calculation: t0,tend,dt,lmin,lmax,nbins,nlobatto,binwidth,tolerances,etc."""
        """ psi0 (list): spectral representation of the initial wavefunction """
        """ method (str):       'static' - solve time-independent Schrodinger equation at time t=t0
                                'dynamic_direct' - propagate the wavefunction with direct exponentiation of the Hamiltonian
                                'dynamic_lanczos' - propagate the wavefunction with iterative method (Lanczos: Krylov subspace method)
            basis (str):        'prim' - use primitive basis
                                'adiab' - use adiabatic basis from t=t0
            ini_state (dict):   {'method': manual,file,calc 'filename':filename}
                                    'method': manual - place the initial state manually in the primitive basis
                                              file - read the inititial state from file given in primitive basis
                                              calc - calculate initial state from given orbitals, by projection onto the primitive basis (tbd)
                                    'filename': filename - filename containing the initial state

            potential (str):    name of the potential energy function (for now it is in an analytic form)
            field_type (str):   type of field: analytic or numerical
            field (str):        name of the electric field function used (it can further read from file or return analytic field)
            scheme (str):       quadrature scheme used in angular integration
        """

        if params['method'] == 'static':

            print("Setting up mixed DVR-Spectral basis set")
            print("\n")

            print("    1) Generating radial grid")
            print("\n")
            #we need to create rgrid only once, i.e. we use a static grid
            rbas = radbas(params['nlobatto'], params['nbins'], params['binwidth'], params['rshift'])
            rgrid = rbas.r_grid()
            nlobatto = params['nlobatto']
            #rbas.plot_chi(0.0,params['nbins'] * params['binwidth'],1000)


            print("    2) Generating mapping of basis set indices")
            print("\n")
            mymap = mapping(int(params['lmin']), int(params['lmax']), int(params['nbins']), int(params['nlobatto']))
            maparray, Nbas = mymap.gen_map()
            params['Nbas'] = Nbas
            print("Nbas = "+str(Nbas))


            hamiltonian = matelem.hmat(params,0.0,rgrid,maparray)
            evals, coeffs, hmat = hamiltonian.calc_hmat()
            #print(type(coeffs))
            #print(np.shape(coeffs))
            rbas.plot_wf_rad(0.0,params['nbins'] * params['binwidth'],1000,coeffs,maparray,rgrid)
            return hmat


        elif params['method'] == 'dynamic_direct':

            """ To do: 1) cleanup; 2) better plotting and saving utilities; 3) fix bug in KEO -> static problem; 4) run orbital + CPL  """

            print("Solving TDSE with a direct exponentiation method")


            print("Setting up mixed DVR-Spectral basis set")
            print("\n")

            print("    1) Generating radial grid")
            print("\n")
            #we need to create rgrid only once, i.e. we use a static grid
            rbas = radbas(params['nlobatto'], params['nbins'], params['binwidth'], params['rshift'])
            rgrid = rbas.r_grid()
            nlobatto = params['nlobatto']
            #rbas.plot_chi(0.0,params['nbins'] * params['binwidth'],1000)


            print("    2) Generating mapping of basis set indices")
            print("\n")
            mymap = mapping(int(params['lmin']), int(params['lmax']), int(params['nbins']), int(params['nlobatto']))
            print("Number of bins = " + str(params['nbins']))
            maparray, Nbas = mymap.gen_map()
            params['Nbas'] = Nbas
            print("Nbas = "+str(Nbas))

            print("Setting up propagation parameters")
            print("\n")
            print("    1) time-grid")
            print("\n")
            timegrid = np.linspace(params['t0'],params['tmax'],int((params['tmax']-params['t0'])/params['dt']+1), endpoint = True)
            dt = params['dt']

            print("    2) initial wavefunction")
            print("\n")
            psi = np.zeros(Nbas,dtype=complex) #spectral representation of the wavefunciton
            #HERE convert psi0 to an instance of the wavefunction class

            for ielem_map,elem_map in enumerate(maparray):
                for ielem_wf0,elem_wf0 in enumerate(psi0):
                    if elem_map[0] == elem_wf0[0] and elem_map[1] == elem_wf0[1] and elem_map[2] == elem_wf0[2] and elem_map[3] == elem_wf0[3]:
                        psi[ielem_map] = elem_wf0[4]
            #normalize the wavefunction: we can move this method to the wavefunction class
            psi[:] /= np.sqrt( np.sum( np.conj(psi) * psi ) )

            print("    3) wavepacket parameters")
            print("\n")
            fname=open(params['wavepacket_file'],'w')
            wavepacket = np.zeros((len(timegrid),Nbas),dtype=complex) #array storing the wavepacket coefficients in order given by the mapping function


            if params['plot_modes']["single_shot"] == True:     
                """ Initialize single_shot plots """   
                # radial plot grid    
                r = np.linspace(params['rshift'],params['nbins']*params['binwidth'],1000,True,dtype=float)

                # polar plot grid
                xr = np.arange(0, 1, 0.01)
                gridtheta = 2 * np.pi * xr
                phi0 = 0.0 #np.pi/4 #at a fixed phi value
                theta0 = 0.0#np.pi/4 #at a fixed phi value
                r0 = 1.0
                #generate Gauss-Lobatto quadrature
                xlobatto=np.zeros(nlobatto)
                wlobatto=np.zeros(nlobatto)
                xlobatto,wlobatto=rbas.gauss_lobatto(nlobatto,14)
                wlobatto=np.array(wlobatto)
                xlobatto=np.array(xlobatto) # convert back to np arrays

            """ Initialize the interaction matrix """
            intmat = np.zeros(( Nbas , Nbas ), dtype = complex)

            """ initialize Hamiltonian """
            start_time = time.time()    
            hamiltonian = matelem.hmat(params,0.0,rgrid,maparray)
            evals, coeffs, hmat = hamiltonian.calc_hmat()
            end_time = time.time()
            print("The execution time for building the field-free Hamiltonian is: " + str(end_time-start_time))

            """Diagonal H0 matrix for testing TDSE solver"""
            #hmat= np.zeros((Nbas, Nbas), float)
            #np.fill_diagonal(hmat, 0.0)

            """ create field object """

            Elfield = Field(params)
            #Fvec = Elfield.gen_field()


            print("==========================")
            print("==wavepacket propagation==")
            print("==========================")

            for itime, t in enumerate(timegrid): #only high-level operations
                print("t = " + str("%10.1f"%t))

                """ calculate interaction matrix at time t """
                """ choose which function type: cartesian or spherical is used to evaluate matrix elements of dipole, before the loop begins"""

                start_time = time.time()    
                vint = hamiltonian.calc_intmat(t,intmat,Elfield.gen_field(t)) #we want to avoid initializing intmat every time-step. So we do it only once and pass it to the function. 

                """matrix exponentiation"""
                umat = expm( -1.0j * (hmat + vint) * dt ) #later: call iterative eigensolver function
                """action of the evolution operator"""
                wavepacket[itime,:] = np.dot( umat , psi )
                """update the wavefunction"""
                psi[:] = wavepacket[itime,:] 
                end_time = time.time()

                print("The execution time for single-step wavefunction propagation is: " + str(end_time-start_time))
                
                #call saving function
                #if requested: method for saving the wavepacket: switch it into a method of psi class?             """save the wavepacket in a file"""
                fname.write(' '.join(["%15.8f"%t])+' '.join(["  %15.8f"%np.abs(item) for item in psi])+"\n")

                """
                fname.write('{:10.3f}'.format(t)+" ".join('{:16.8e}'.format(Psi[i].real)+'{:16.8e}'.format(Psi[i].imag) for i in range(0,Ntotal))+\
                    '{:10.3f}'.format(wfnorm)+"\n")
                """

                
            """ ========= POST-PROCESSING ========= """ 

            print("=====================================")
            print("==post-processing of the wavepacket==")
            print("=====================================")


            """ calculate timegrid indices at which the relevant observables are plotted"""


            """    params['plot_modes'] = {"single_shot": True, "animation": False}

                    params['plot_types'] = { "radial": True,
                             "angular": True,
                             "r-radial_angular": True, 
                             "k-radial_angular": True} #decide which of the available observables you wish to plot

                    params['plot_controls'] = { "plotrate": 1, 
                                "plottimes": [10.0,50.0,76.0,120.0],
                                "save_static": True,
                                "save_anim": False,
                                "show_static": True,
                                "show_anim": False 
                                "static_filename": "obs"
                                "animation_filename":"anim_obs"}
            """
            iplot = []
            for index,item in enumerate(params['plot_controls']["plottimes"]):

                if int(np.float64(1.0/24.188)* item/dt) > len(timegrid):
                    print("removing time: "+str(item)+" from plotting list. Time exceeds the propagation time-grid!")
                else:
                    iplot.append(int(np.float64(1.0/24.188)* item/dt))

            print("Final list of plot times:")
            print(iplot)


            for itime, t in enumerate(timegrid): 
                print("t = " + str("%10.1f"%t))

                if params['plot_modes']["single_shot"] == True:
                    for ielem in iplot:
                        if itime == ielem:
                            psi[:] = wavepacket[itime,:]
                            #plot format
                            plot_format = [int(params['plot_types']["radial"]),
                             int(params['plot_types']["angular"]),
                             int(params['plot_types']["r-radial_angular"]),
                            int(params['plot_types']["k-radial_angular"])]
                            self.plot_snapshot(psi,params['plot_controls'],plot_format,t,rgrid,wlobatto,r,maparray,phi0,theta0,r0,rbas)



    def plot_snapshot(self,psi,plot_controls,plot_format,t,rgrid,wlobatto,r,maparray,phi0,theta0,r0,rbas):
        if sum(plot_format) == 4:
            print("plotting all static graphs")

            fig = plt.figure(figsize=(10, 5), dpi=300, constrained_layout=True)

            spec = gridspec.GridSpec(ncols=2, nrows=2,figure=fig)
            axradang_r = fig.add_subplot(spec[0, 0])
            axradang_k = fig.add_subplot(spec[1, 1])
            axrad = fig.add_subplot(spec[0, 1])
            axang = fig.add_subplot(spec[1, 0])
            y = np.zeros(len(r),dtype=complex)
            #plt.tight_layout()
            #================ radial wf ===============#
            axrad.set_xlabel('r (a.u.)')
            axrad.set_ylabel(r'$r|\psi(r,\theta_0,\phi_0)|^2$')
            axrad.set_xlim( params['rshift'] , params['binwidth'] * params['nbins'] + params['rshift'] )
            #axrad.set_ylim(0.0, 1.0)
            """ 1) Radial wavepacket at fixed phi and theta """
            for ielem,elem in enumerate(maparray):
                y +=    psi[ielem] * rbas.chi(elem[2],elem[3],r,rgrid,wlobatto) * sph_harm(elem[1], elem[0],  phi0, theta0)


            line_rad, = axrad.plot(r, r*np.abs(y), label="time = "+str(t/time_to_au)+ " as", color='black', marker='.', linestyle='solid', linewidth=2)
            plt.legend()

            if plot_controls["show_static"] == True:
                plt.show()   
    

            """  2) Populations of states """       
            """if int(itime)%params['plotrate']== 0:
                for i in range(Nbas):
                    #plt.plot(timegrid,wavepacket[:,i].real)
                    #plt.plot(timegrid,wavepacket[:,i].imag)
                    plt.plot(timegrid,np.sqrt(wavepacket[:,i].real**2+wavepacket[:,i].imag**2),'-+',label=str(i))
                plt.xlabel('t/a.u.')
                plt.ylabel('populations')
                plt.legend()
                plt.show()
            """
            """ 3) Polar plot or (r,theta) at fixed phi """
            """
            angwf = np.zeros(len(gridtheta),dtype=complex)

            if int(itime)%params['plotrate'] == 0:
                for ielem,elem in enumerate(maparray):
                    angwf[:] += psi[ielem]  * sph_harm(elem[1], elem[0],  phi0, gridtheta)# * rbas.chi(elem[2],elem[3],r0,rgrid,wlobatto) 

                
                ax = plt.subplot(111, projection='polar')
                ax.plot(gridtheta, np.abs(angwf), 'r+', label="angular-radial wf",linewidth=3)
                #ax.set_rmax(1)
                ax.set_rticks([0.5, 1])  # Less radial ticks
                ax.grid(True)
                plt.legend()        
                ax.set_title("Radial-angular representation of photo-electron wavepacket", va='bottom')
                plt.show()
            """


    def plot_mat(self,mat):
        """ plot 2D array with color-coded magnitude"""
        fig, ax = plt.subplots()

        im, cbar = self.heatmap(mat, 0, 0, ax=ax, cmap="gnuplot", cbarlabel="Hij")

        fig.tight_layout()
        plt.show()

    def heatmap(self, data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        """ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)"""

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        #ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        #ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def gen_psi0(self,params):
        psi0_mat = []
        """ generate initial conditions for TDSE """

        """returns: psi0_list (list: Nx4): l(int),m(int),i(int),n(int),c_lmin(complex128) """
        """ ordering l,m,i,n in descening hierarchy"""

        if params['ini_state'] == "spectral_manual":
            print("defining initial wavefunction manually \n")
            psi0_mat.append([0,0,0,0,1.0+0.0j])
            #psi0_mat.append([0,0,0,1,1.0+0.0j])
            #psi0_mat.append([0,0,0,2,1.0+0.0j])


            print("initial wavefunction:")
            print(psi0_mat)
            return psi0_mat

        elif params['ini_state'] == "spectral_file":
            print("reading initial wavefunction from file \n")
            fl = open(params['ini_state_file_coeffs'],'r')

            for line in fl:
                words = line.split()
                # columns: l, m, i, n, coef

                l = int(words[0])
                m = int(words[1])
                i = int(words[2])
                n = int(words[3])
                coeff = float(words[4])
                psi0_mat.append([l,m,i,n,coeff])

            #normalize the w-f

            norm = 0.0
            for ind in range(len(psi0_mat)):
                norm+=np.conj(psi0_mat[ind][4]) * psi0_mat[ind][4]

            sum = 0 
            for ind in range(len(psi0_mat)):
                psi0_mat[ind][4] /= np.sqrt(norm)
                sum += np.conj(psi0_mat[ind][4]) * psi0_mat[ind][4]
            print(psi0_mat)
            return psi0_mat

        elif params['ini_state'] == "grid_1d_rad":
            print("generating initial wavefunction by projection of a given function onto our basis \n")
            print("1D radial coordinate only \n")
          
            mymap = mapping(int(params['lmin']), int(params['lmax']), int(params['nbins']), int(params['nlobatto']))
            maparray, Nbas = mymap.gen_map()
            #l,m,i,n

            #generate requested quadrature
            if params['ini_state_quad'][0] == "Gauss-Laguerre":
                Nquad = params['ini_state_quad'][1]
                x,w = roots_genlaguerre(Nquad ,1)
                alpha = 1 #parameter of the G-Lag quadrature
                inv_weight_func = lambda r: r**(-alpha)*np.exp(r)
            elif params['ini_state_quad'][0] == "Gauss-Hermite":
                Nquad = params['ini_state_quad'][1]
                x,w = roots_hermite(Nquad)
                inv_weight_func = lambda r: np.exp(r**2)


            #generate gauss-Lobatto global grid
            rbas = radbas(params['nlobatto'], params['nbins'], params['binwidth'], params['rshift'])
            nlobatto = params['nlobatto']
            nbins = params['nbins']
            rgrid  = rbas.r_grid()
            #generate Gauss-Lobatto quadrature
            xlobatto=np.zeros(nlobatto)
            wlobatto=np.zeros(nlobatto)
            xlobatto,wlobatto=rbas.gauss_lobatto(nlobatto,14)
            wlobatto=np.array(wlobatto)
            xlobatto=np.array(xlobatto) # convert back to np arrays


            for elem in maparray:
                coeff = 0.0
                for k in range(Nquad):
                    coeff += w[k] * rbas.chi(elem[2],elem[3],x[k],rgrid,wlobatto)  * self.psi01d(x[k]) * inv_weight_func(x[k])
            

                print("%5d"%float(elem[0])+"  %5d"%float(elem[1])+"  %5d"%float(elem[2])+"  %5d"%float(elem[3])+"  %6.3f"%float(coeff))
                psi0_mat.append([ elem[0], elem[1], elem[2], elem[3], coeff ])

            #print(psi0_mat)

            #plot the test function and the projected function


            r = np.linspace(0.01,100.0,1000,True,dtype=float)
            y = np.zeros(len(r))

            psi0_mat = np.asarray(psi0_mat)
            counter = 0
            for elem in maparray:
                y[:] +=   psi0_mat[counter,4] * rbas.chi(elem[2],elem[3],r,rgrid,wlobatto)

                counter +=1
            y /= np.sqrt(np.sum(psi0_mat[:,4] * psi0_mat[:,4]))



            plt.xlabel('r/a.u.')
            plt.ylabel('orbitals')
    
            plt.plot(r, r*y/max(np.abs(y))) 
            plt.plot(r, r*self.psi01d(r)/max(np.abs(self.psi01d(r))))
            plt.show()   


            #rbas.plot_chi(0.01,100,1000)

            val = 0.0
            for i in range(len(r)):
                val += ( y[i]/max(np.abs(y)) - self.psi01d(r[i])/max(np.abs(self.psi01d(r))) ) **2
            rmsd = np.sqrt( val/len(r) )  
        
            print("RMSD = " + str(rmsd))
            fmax = max(np.abs(self.psi01d(r)))
            print(fmax)
            print("RMSD relative to maximum function value = " + str(rmsd / fmax * 100)+" %")
            exit()

        elif params['ini_state'] == "grid_2d_sph":  
            print("generating initial wavefunction by projection of a given function onto our basis \n")
            print("2D spherical coordinates only \n")
          
            mymap = mapping(int(params['lmin']), int(params['lmax']), int(params['nbins']), int(params['nlobatto']))
            maparray, Nbas = mymap.gen_map()
            #l,m,i,n

            myscheme = quadpy.u3.schemes["lebedev_025"]()

            """
            Symbols: 
                    theta_phi[0] = theta in [0,pi]
                    theta_phi[1] = phi  in [-pi,pi]
                    sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) means that we put the phi angle in range [0,2pi] and the  theta angle in range [0,pi] as required by the scipy special funciton sph_harm
            """

            for elem in maparray:
                coeff = 0.0 + 0.0j
                coeff = myscheme.integrate_spherical( lambda theta_phi: np.conjugate(sph_harm(elem[1], elem[0],  theta_phi[1]+np.pi, theta_phi[0])) * self.psi02d(theta_phi[0],theta_phi[1]+np.pi)  )
                print("%5d"%float(elem[0])+"  %5d"%float(elem[1])+ '%10.5f %10.5fi' % (coeff.real, coeff.imag))
                psi0_mat.append([ elem[0], elem[1], coeff ])

            #plot the test function and the projected function
            r = np.arange(0, 1, 0.01)
            theta = 2 * np.pi * r
            phi0 = np.pi/2 #at a fixed phi value

            y = np.zeros(len(theta),dtype=float)
            psi0_mat = np.asarray(psi0_mat)
            counter = 0
            for elem in maparray:
                y[:] += np.real(psi0_mat[counter,2]) * np.real(sph_harm(elem[1], elem[0],  phi0, theta))
                counter +=1
            
            ax = plt.subplot(111, projection='polar')
            ax.plot(theta, y/max(np.abs(y)), label="projected wf",linewidth=3)
            ax.plot(theta, np.real(self.psi02d(theta, phi0) )/max(np.abs(self.psi02d(theta, phi0))), label="original wf")
            ax.set_rmax(1)
            ax.set_rticks([0.5, 1])  # Less radial ticks
            ax.grid(True)
            plt.legend()        
            ax.set_title("Comparison of original and projected spherical funcitons", va='bottom')
            plt.show()

           

            #rbas.plot_chi(0.01,100,1000)

            """ val = 0.0
            for i in range(len(r)):
                val += ( y[i]/max(np.abs(y)) - self.psi01d(r[i])/max(np.abs(self.psi01d(r))) ) **2
            rmsd = np.sqrt( val/len(r) )  
        
            print("RMSD = " + str(rmsd))
            fmax = max(np.abs(self.psi01d(r)))
            print(fmax)
            print("RMSD relative to maximum function value = " + str(rmsd / fmax * 100)+" %")"""

            """#save function in wf0grid.txt
            fl_out = open( params['ini_state_file_grid'],'w')
            for i in range(len(nodes)):
            fl_out.write("%6.3f"%float(nodes[i][0])+"  %6.3f"%float(nodes[i][1])+"  %6.3f"%float(nodes[i][2])+" %6.3f"%self.gen_test_hydrogen_wf(nodes[i][0],nodes[i][1], nodes[i][2])+"\n")


            # readf= function on a grid from file
            fl = open(params['ini_state_file_grid'],'r')
            iniwf = []
            for line in fl:
                words = line.split()
                # columns: l, m, i, n, coef

            r = float(words[0])
            theta = float(words[1])
            phi = float(words[2])
            wf = float(words[3])
            iniwf.append([r,theta,phi,wf]) #initial wavefunction on a grid


            input: complex wavefunction on r,theta,phi grid
            return: set of coefficients in l,m,i,n basis
             """

        elif params['ini_state'] == "grid_3d":  
            print("generating initial wavefunction by projection of a given function onto our basis \n")
            print("3D spherical+radial coordinates\n")
          
            mymap = mapping(int(params['lmin']), int(params['lmax']), int(params['nbins']), int(params['nlobatto']))
            maparray, Nbas = mymap.gen_map()
            #l,m,i,n

            myscheme = quadpy.u3.schemes["lebedev_025"]()

            #generate requested quadrature
            if params['ini_state_quad'][0] == "Gauss-Laguerre":
                Nquad = params['ini_state_quad'][1]
                x,w = roots_genlaguerre(Nquad ,1)
                alpha = 1 #parameter of the G-Lag quadrature
                inv_weight_func = lambda r: r**(-alpha)*np.exp(r)
            elif params['ini_state_quad'][0] == "Gauss-Hermite":
                Nquad = params['ini_state_quad'][1]
                x,w = roots_hermite(Nquad)
                inv_weight_func = lambda r: np.exp(r**2)


            #generate gauss-Lobatto global grid
            rbas = radbas(params['nlobatto'], params['nbins'], params['binwidth'], params['rshift'])
            nlobatto = params['nlobatto']
            nbins = params['nbins']
            rgrid  = rbas.r_grid()
            #generate Gauss-Lobatto quadrature
            xlobatto=np.zeros(nlobatto)
            wlobatto=np.zeros(nlobatto)
            xlobatto,wlobatto=rbas.gauss_lobatto(nlobatto,14)
            wlobatto=np.array(wlobatto)
            xlobatto=np.array(xlobatto) # convert back to np arrays

            """
            Symbols: 
                    theta_phi[0] = theta in [0,pi]
                    theta_phi[1] = phi  in [-pi,pi]
                    sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) means that we put the phi angle in range [0,2pi] and the  theta angle in range [0,pi] as required by the scipy special funciton sph_harm
            """

            for elem in maparray:
                coeff = 0.0 + 0.0j
               
                coefftemp = lambda r: myscheme.integrate_spherical( lambda theta_phi: np.conjugate(sph_harm(elem[1], elem[0],  theta_phi[1]+np.pi, theta_phi[0])) * self.psi03d(r,theta_phi[0],theta_phi[1]+np.pi)  )

                for k in range(Nquad):
                    coeff += w[k] * rbas.chi(elem[2],elem[3],x[k],rgrid,wlobatto)  * coefftemp(x[k]) * inv_weight_func(x[k])
                    

                print("%5d"%float(elem[0])+"  %5d"%float(elem[1])+"  %5d"%float(elem[2])+"  %5d"%float(elem[3])+ '%10.5f %10.5fi' % (coeff.real, coeff.imag))
                psi0_mat.append([ elem[0], elem[1], elem[2], elem[3], coeff ])


            #plot the test function and the projected function
            

            #spherical part
            xr = np.arange(0, 1, 0.01)
            theta = 2 * np.pi * xr
            phi0 = np.pi/4 #at a fixed phi value
            r0 = 1.0

            y = np.zeros(len(theta),dtype=complex)
            psi0_mat = np.asarray(psi0_mat)
            counter = 0
            for elem in maparray:
                y[:] += psi0_mat[counter,4] * sph_harm(elem[1], elem[0],  phi0, theta) * rbas.chi(elem[2],elem[3],r0,rgrid,wlobatto) 
                counter +=1
            
            ax = plt.subplot(111, projection='polar')
            ax.plot(theta, np.abs(y)/max(np.abs(y)), 'r+', label="projected wf",linewidth=3)
            ax.plot(theta, np.abs(self.psi03d(r0,theta, phi0) )/max(np.abs(self.psi03d(r0,theta, phi0))), label="original wf")
            ax.set_rmax(1)
            ax.set_rticks([0.5, 1])  # Less radial ticks
            ax.grid(True)
            plt.legend()        
            ax.set_title("Comparison of original and projected spherical funcitons", va='bottom')
            plt.show()

            #radial part
            r = np.linspace(0.01,float(params['binwidth'] * params['nbins']),1000,True,dtype=float)
            y = np.zeros(len(r),dtype=complex)

            counter = 0
            for elem in maparray:
                y[:] +=   psi0_mat[counter,4] * rbas.chi(elem[2],elem[3],r,rgrid,wlobatto) * sph_harm(elem[1], elem[0],  phi0, phi0)

                counter +=1
            y /= np.sqrt(np.sum(psi0_mat[:,4] * psi0_mat[:,4]))



            plt.xlabel('r/a.u.')
            plt.ylabel('orbitals')
    
            plt.plot(r, r * np.abs(y)/max(np.abs(y)), 'r+',label="projected wf" ) 
            plt.plot(r, r * np.abs(self.psi03d(r,phi0, phi0))/max(np.abs(self.psi03d(r,phi0, phi0))) ) 
            plt.legend()     
            plt.show()   

        if params['ini_state'] == "eigenvec":
            print("defining initial wavefunction by diagonalization of static Hamiltonian \n")
            temp = params['nbins'] 
            params['nbins'] = params['nbins_iniwf']

            map_iniwf = mapping(int(params['lmin']), int(params['lmax']), int(params['nbins']), int(params['nlobatto']))
            maparray_iniwf, Nbas0 = map_iniwf.gen_map()
            params['Nbas'] = Nbas0
            print("Number of basis functions used to diagonalize the static Hamiltonian = " + str(Nbas0))
            #l,m,i,n

            #generate gauss-Lobatto global grid
            rbas = radbas(params['nlobatto'], params['nbins'], params['binwidth'], params['rshift'])
            nlobatto = params['nlobatto']
            nbins = params['nbins']
            rgrid  = rbas.r_grid()
            #generate Gauss-Lobatto quadrature
            xlobatto=np.zeros(nlobatto)
            wlobatto=np.zeros(nlobatto)
            xlobatto,wlobatto=rbas.gauss_lobatto(nlobatto,14)
            wlobatto=np.array(wlobatto)
            xlobatto=np.array(xlobatto) # convert back to np arrays
            phi0 = 0.0 #
            theta0 = 0.0 #
            r0 = 1.0

            """ initialize Hamiltonian """
            hamiltonian0 = matelem.hmat(params,0.0,rgrid,maparray_iniwf)
            evals, coeffs, hmat = hamiltonian0.calc_hmat()

            for ielem, elem in enumerate(maparray_iniwf):
                psi0_mat.append([ elem[0], elem[1], elem[2], elem[3], coeffs[ielem,int(params['eigenvec_id'])]])

            psi0_mat = np.asarray(psi0_mat)
            #print("eivenvectors of static Hamiltonian")
            #print(coeffs)

            #radial part
            r = np.linspace(params['rshift'],float(params['binwidth'] * params['nbins_iniwf']),1000,True,dtype=float)
            y = np.zeros(len(r),dtype=complex)
            for ielem,elem in enumerate(maparray_iniwf):
                y[:] +=   coeffs[ielem,int(params['eigenvec_id'])] * rbas.chi(elem[2],elem[3],r,rgrid,wlobatto) * sph_harm( elem[1], elem[0],  theta0, phi0)
            y /= np.sqrt(np.sum(coeffs[:,int(params['eigenvec_id'])] * coeffs[:,int(params['eigenvec_id'])]))

            plt.xlabel('r/a.u.')
            plt.ylabel('orbitals')    
            plt.plot(r, r * np.abs(y)/max(np.abs(y)), 'r+',label="calculated eigenfunction" ) 
            plt.plot(r, r * np.abs(self.psi03d(r,phi0, phi0))/max(np.abs(self.psi03d(r,phi0, phi0))) ) 
            plt.legend()     
            plt.show()  

            if params['save_ini_wf'] == True:
                fname=open(params['wavepacket_file'],'w')
                print("saving the initial wavefunction into file")
                fname.write(' '.join(["  %15.8f"%np.abs(item) for item in psi0_mat])+"\n")
            params['nbins'] = temp 
        return psi0_mat

    def gen_hydrogen_wf(self,n,l,m,r,theta,phi):
        Rnl = (2*r/n)**l * np.exp(-r/n) * genlaguerre(n-l-1,2*l+1)(2*r/n)
        Ylm = sph_harm(m,l, phi,theta)
        return Rnl, Ylm, Rnl * Ylm

    def psi0(self,r,theta,phi):
        """ return custom function in the hydrogen atom orbital basis. It can be multi-center molecular orbital"""
        h_radial = lambda r,n,l: (2*r/n)**l * np.exp(-r/n) * genlaguerre(n-l-1,2*l+1)(2*r/n)
        f = 0.0
        f +=   h_radial(r,3,0) * sph_harm(0,0, phi,theta)
        return f

    def psi01d(self,r):
        """ return custom function in 1d in the hydrogen atom orbital basis"""
        h_radial = lambda r,n,l: (2/n)**(3/2)*np.sqrt(factorial(n-l-1)/(2*n*factorial(n+l)))*(2*r/n)**l * np.exp(-r/n) * genlaguerre(n-l-1,2*l+1)(2*r/n)
        f = 0.0
        f +=   h_radial(r,3,1)
        f /= np.sqrt(float(1.0)) 
        return f

    def psi02d(self,theta,phi):
        """ return custom function in 2d in spherical coordintates"""
        fsph = lambda theta,phi,l,m: sph_harm(m,l,phi,theta)
        f = 0.0
        f +=  1/(np.cos(theta)**2-2.0)#fsph(theta,phi,2,2)
        f /= np.sqrt(float(1.0)) 
        return f

    def psi03d(self,r,theta,phi):
        """ return custom function in 3D"""
        fsph = lambda theta,phi,l,m: sph_harm(m,l,phi,theta)
        Rnl = lambda r,n,l: (2/n)**(3/2)*np.sqrt(factorial(n-l-1)/(2*n*factorial(n+l)))*(2*r/n)**l * np.exp(-r/n) * genlaguerre(n-l-1,2*l+1)(2*r/n)
        f = 0.0
        f +=  Rnl(r,2,0) * fsph(theta,phi,0,0) #1/(r * np.cos(theta)**2+2.0)
        # 2p0 orbital: Rnl(r,2,1) * fsph(theta,phi,1,0)
        # 1s orbital:  Rnl(r,1,0) * fsph(theta,phi,0,0)
        f /= np.sqrt(float(1.0)) 

        return f

    def plot_hydrogen_wfs(self):

        """ generate the radial part """
        #Test radial function:
        h_radial = lambda r,n,l: (2*r/n)**l * np.exp(-r/n) * genlaguerre(n-l-1,2*l+1)(2*r/n)
        
        #3D grid
        radgrid = np.linspace(0.0,20.0,100)
        thetagrid = np.linspace(0,np.pi,20)
        phigrid  = np.linspace(0,2*np.pi,20)
        grid3d = np.meshgrid(radgrid,thetagrid,phigrid)
        nodes = list(zip(*(dim.flat for dim in grid3d)))
        nodes = np.asarray(nodes)

        for node in nodes:
            print(node)
        plt.plot(nodes[:,0],h_radial(nodes[:,0],2,0)**2 * nodes[:,0]**2)
        plt.show()


        """generate the angular part"""
        theta_2d, phi_2d = np.meshgrid(thetagrid, phigrid)
        xyz_2d = np.array([np.sin(theta_2d) * np.sin(phi_2d), np.sin(theta_2d) * np.cos(phi_2d), np.cos(theta_2d)]) #2D grid of cartesian coordinates
        colormap = cm.ScalarMappable( cmap=plt.get_cmap("cool") )
        colormap.set_clim(-.45, .45)
        limit = .5

        plt.figure()
        ax = plt.gca(projection = "3d")
        #Test angular function
        Y_lm = sph_harm(1,0, phi_2d,theta_2d)
        print(np.shape(Y_lm))
        #r = np.abs(Y_lm.real)*xyz_2d #calculate a point in 3D cartesian space for each value of spherical harmonic at (theta,phi)
        #r = np.abs(iniwf(2.0,theta_2d, phi_2d))*xyz_2d #calculate a point in 3D cartesian space for each value of spherical harmonic at (theta,phi)  
        r = self.test_hydrogen_wf(1.0, theta_2d, phi_2d) * xyz_2d 
        ax.plot_surface(r[0], r[1], r[2], facecolors=colormap.to_rgba(Y_lm.real), rstride=1, cstride=1)
        ax.set_xlim(-limit,limit)
        ax.set_ylim(-limit,limit)
        ax.set_zlim(-limit,limit)
        plt.show()

    def Lanczos(self,Psi, t, HMVP, m, Ntotal, timestep,Ham0):
        Psi1=Psi.copy()
        #Psi is the input vector of coefficients of size N
        # run Lanczos algorithm to calculate basis of Krylov space
        H=[] #full Hamiltonian matrix
        H=ham(Ham0,t,Ntotal)
        #m is the size of the Krylov space
        q, v=[],[]
        Hm=zeros((m, m)) #Hamiltonian in Krylov space
        norms=zeros(m)
        
        H=ham(Ham0,t,Ntotal)
        
        for j in range(0, m):
            norms[j]=norm(Psi1) #beta_j
            # q builds up the basis of Krylov space
            q=Psi1/norms[j] #q
            
            Psi1=HMVP(H,q) #r=H*q
            
            if j>0:
                Hm[j-1, j]=Hm[j, j-1]=vdot(v, Psi1).real
                Psi1-=Hm[j-1, j]*v
                
            Hm[j, j]=vdot(q, Psi1).real #alpha_0
            Psi1-=Hm[j, j]*q #new r:=r-alpha0*q
            v, q=q, v
        # diagonalize A
        dm, Zm=eigh(Hm) #dm are eigenvalues of Hm and Zm are eigenvectors of size m.
        
        # calculate matrix exponential in Krylov space
        zdprod=dot(Zm, dot(diag(exp(-1j*dm*timestep)), Zm[0, :]*norms[0]))
        
        # run Lanczos algorithm 2nd time to transform result into original space
        
        Psi1, Psi=Psi, zeros_like(Psi)
        
        for j in range(0, m):
            q=Psi1/norms[j]
            np.add(Psi, zdprod[j]*q, out=Psi, casting="unsafe")
        # Psi+=zdprod[j]*q
            
            Psi1=HMVP(H,q)
            
            if j>0:
                Hm[j-1, j]=Hm[j, j-1]=vdot(v, Psi1).real
                Psi1-=Hm[j-1, j]*v
            Hm[j, j]=vdot(q, Psi1).real
            Psi1-=Hm[j, j]*q
            v, q=q, v
        return Psi

    def mvp(self,H,q): #for now we are passing the precomuted Hamiltonian matrix into the MVP routine, which is inefficient.
        # I will have to code MVP without invoking the Hamiltonian matrix.
        return np.matmul(H,q)



if __name__ == "__main__":      


    params = {}
    
    """====basis set parameters===="""
    params['nlobatto'] = 8
    params['nbins'] = 5 #bug when nbins > nlobatto  
    params['binwidth'] = 3.0
    params['rshift'] = 1e-5 #rshift must be chosen such that it is non-zero and does not cover significant probability density region of any eigenfunction.
    params['lmin'] = 0
    params['lmax'] = 1
    
    """====runtime controls===="""
    params['method'] = "dynamic_direct" #static: solve time-independent SE for a given potential; dynamic_direct, dynamic_lanczos
    params['basis'] = "prim" # or adiab
    params['potential'] = "hydrogen" # 1) diagonal (for tests); 2) hydrogen
    params['scheme'] = "lebedev_019" #angular integration rule
    params['t0'] = 0.0 
    params['tmax'] = 100.0 
    params['dt'] = 10.0 
    time_units = "as"


    """===== post-processing and analysis ====="""

    params['wavepacket_file'] = "wavepacket.dat" #filename into which the time-dependent wavepacket is saved
    params['plot_modes'] = {"single_shot": True, "animation": False}

    params['plot_types'] = { "radial": True,
                             "angular": True,
                             "r-radial_angular": True, 
                             "k-radial_angular": True} #decide which of the available observables you wish to plot

    params['plot_controls'] = { "plotrate": 1, 
                                "plottimes": [10.0,50.0,76.0,120.0],
                                "save_static": True,
                                "save_anim": False,
                                "show_static": True,
                                "show_anim": False, 
                                "static_filename": "obs",
                                "animation_filename": "anim_obs"}

    """ plotrate : rate of plotting observables in timestep units in animated plots
        plottimes: times (in time_units) at which we plot selected observables in a static graph
        save_static: save single shot plots to appropriate files (named separately for each plottime)
        save_anim: save animation in a file
        show_static: show single shot plots during the analysis
        show_anim: show animation at the end of analysis
        static_filename: name of the file into which the snapshots will be saved
        animation_filename: name of the file into which animations will be saved
    """



    """====initial state====""" 
    params['ini_state'] = "eigenvec" #spectral_manual, spectral_file, grid_1d_rad, grid_2d_sph,grid_3d,solve (solve static problem in Lobatto basis), eigenvec (eigenfunciton of static hamiltonian)
    params['ini_state_quad'] = ("Gauss-Laguerre",80) #quadrature type for projection of the initial wavefunction onto lobatto basis: Gauss-Laguerre, Gauss-Hermite
    params['ini_state_file_coeffs'] = "wf0coeffs.txt" # if requested: name of file with coefficients of the initial wavefunction in our basis
    params['ini_state_file_grid'] = "wf0grid.txt" #if requested: initial wavefunction on a 3D grid of (r,theta,phi)
    params['nbins_iniwf'] = 5 #number of bins in a reduced-size grid for generating the initial wavefunction by diagonalizing the static hamiltonian
    params['eigenvec_id'] = 1 #id (in ascending energy order) of the eigenvector of the static Hamiltonian to be used as the initial wavefunction for time-propagation
    params['save_ini_wf'] = False #save initial wavefunction generated with eigenvec option to a file (spectral representation)


    """====field controls===="""
    params['field_form'] = "analytic" #or numerical
    params['field_name'] = "LP" #RCPL, LCPL, ...
    params['field_env'] = "gaussian" #"static_uniform"
    params['omega'] =  400.0 #nm
    #convert to THz:
    vellgt     =  2.99792458E+8 # m/s
    params['omega']= 10**9 *  vellgt / params['omega'] #Hz
    print("Electric field carrier frequency = "+str(params['omega']*1.0e-12)+" THz")
    print("Electric field oscillation period = "+str(1.0e15/params['omega'])+" fs")
    params['omega'] /= 4.13e16 #Hz to a.u.
    frequency_units = "nm" #we later convert all units to atomic unit
    params['E0'] = 1.0e9 #V/cm
    field_units = "V/cm"
    params['sigma'] = 20.0 #as
    params['tau0'] = 20.0 #as #centre of the pulse

    # convert time units to atomic units
    time_to_au = {"as" : np.float64(1.0/24.188)}
    # 1a.u. (time) = 2.418 e-17s = 24.18 as

    # convert frequency units to atomic units
    freq_to_au = {"nm" : np.float64(0.057/800.0)}
    # 1a.u. (time) = 2.418 e-17s = 24.18 as

    # convert electric field from different units to atomic units
    field_to_au = {"debye" : np.float64(0.393456),
                    "V/cm" :  np.float64(1.0/(5.14220652e+9))}

    #unit conversion
    #params = const.convert_units(params)
    time_to_au = time_to_au[time_units]
    params['sigma'] *= time_to_au 
    params['tau0'] *= time_to_au 
    params['t0'] *= time_to_au 
    params['tmax'] *= time_to_au 
    params['dt'] *= time_to_au 

    #freq_to_au = freq_to_au[frequency_units]
    #params['omega'] *= freq_to_au 
    field_to_au = field_to_au[field_units]
    params['E0'] *= field_to_au 
    # 1a.u. (time) = 2.418 e-17s = 24.18 as
    #field strength in a.u. (1a.u. = 5.1422e9 V/cm). For instance: 5e8 V/cm = 3.3e14 W/cm^2
   



    """ We have two options for calculating the potential matrix elements: 
    1) cartesian: we use cartesian elements of the dipole moment operator and the electric field. Lebedev quadratures are used to compute matrix elements
    2) spherical-tensor: sphereical tensor representation of the field and the dipole moment. The matrix elements are computed analytically with 3-j symbols.
    """
   
    params['int_rep_type'] = 'cartesian'


    hydrogen = propagate() #!!!name of the potential should be one of the attributes of the propagate class!!!

    psi0 = hydrogen.gen_psi0(params) #Test = true means: Testing the generation of the initial wavefunction on a grid on the example of analytic hydrogen atoms wavefunctions
    hmat = hydrogen.prop_wf(params,psi0)
    #hydrogen.plot_mat(np.abs(hmat[1:,1:]))
    exit()


    #print(timeit.timeit('hydrogen.prop_wf(method,basis,ini_state,params,potential,field,scheme)',setup='from matelem import hmat', number = 100))  

    
    """0.49842426     -0.12500000]
    [    -0.12500000     -0.05555556]
    [    -0.12500000     -0.03125000]
    [    -0.12500000     -0.02000000]
    [    -0.12480328     -0.01388889]
    [    -0.05555477     -0.01020408]
    [    -0.05555477     -0.00781250]
    [    -0.05555477     -0.00617284]
    [    -0.05549590     -0.00500000]
    [    -0.03071003     -0.00413223]
    [    -0.03071003     -0.00347222]
    [    -0.03071003     -0.00295858]
    [    -0.03052275     -0.00255102]
    [    -0.01221203     -0.00222222]
    [    -0.01221203     -0.00195312]
    [    -0.01221203     -0.00173010]
    [    -0.01102194     -0.00154321]"""
    
    
    """ Test angular convergence of the potential matrix elements """

    """ quad_tol = 1e-6

    potmatrix = potmat(scheme='lebedev_005')
    rin = np.linspace(0.01,100.0,10,endpoint=True)
    print(rin)
    for r in rin:
        print("radial coordinate = "+str(r))
        potmatrix.test_angular_convergence(lmin,lmax,quad_tol,r)
    """

    """ Test print the KEO matrix """
    """keomatrix = keomat()
    keomatrix.calc_mat"""


    """ code profiling
    
    
    cprofile:
    with cProfile.Profile() as pr
        function()

        pr.dump_stats('function')



        python -m cProfile -o myscript.prof myscript.py

        snakewiz myscript.prof

        python -m line_profiles myprofile.prof

        %timeit  #gives statistics
        function()

        %%time
        function() -single call time

        timestart = time.perf_counter()
        function()
        timeend = time.perf_counter()"""