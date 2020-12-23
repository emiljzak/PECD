#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
import sys
import matelem
import timeit
import quadpy
from basis import radbas
from matelem import mapping
from scipy.special import genlaguerre,sph_harm
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

from numpy.polynomial.laguerre import laggauss

class propagate(radbas,mapping):

    def __init__(self):
        pass

    def prop_wf(self,params):
        """ main method for propagating the wavefunction"""
        """params (dict):      dictionary with all relevant numerical parameters of the calculation: t0,tend,dt,lmin,lmax,nbins,nlobatto,binwidth,tolerances,etc."""

        """ method (str):       'static' - solve time-independent Schrodinger equation at time t=t0
                                'direct' - propagate the wavefunction with direct exponentiation of the Hamiltonian
                                'krylov' - propagate the wavefunction with iterative method (Lanczos: Krylov subspace method)
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
            #we need to create rgrid only once, i.e. static grid
            rbas = radbas(params['nlobatto'], params['nbins'], params['binwidth'], params['rshift'])
            rgrid = rbas.r_grid()
            #rbas.plot_chi(0.0,params['nbins'] * params['binwidth'],1000)
            mymap = mapping(int(params['lmin']), int(params['lmax']), int(params['nbins']), int(params['nlobatto']))
            maparray, Nbas = mymap.gen_map()
            params['Nbas'] = Nbas
            print("Nbas = "+str(Nbas))

            #print(type(maparray))
            #print(maparray)

            hamiltonian = matelem.hmat(params,0.0,rgrid,maparray)
            evals, coeffs, hmat = hamiltonian.calc_hmat()
            #print(type(coeffs))
            #print(np.shape(coeffs))
            #rbas.plot_wf_rad(0.0,params['nbins'] * params['binwidth'],1000,coeffs,maparray,rgrid)
            return hmat

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


    def ini_wf(self,params,test=False):
        psi0 = []
        """ generate initial conditions for TDSE """
        """psi0 (array: Nx4): l(int),m(int),i(int),n(int),s_xi(complex128) """
        """ ordering n,i,l,m major"""

        if params['ini_state'] == "manual":
            print("defining initial wavefunction manually \n")
            psi0.append([0,0,0,0,1.0+0.0j])
            print("initial wavefunction:")
            print(psi0)

        elif params['ini_state'] == "file":
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
                psi0.append([l,m,i,n,coeff])

            #normalize the w-f

            norm = 0.0
            for ind in range(len(psi0)):
                norm+=np.conj(psi0[ind][4]) * psi0[ind][4]

            sum = 0 
            for ind in range(len(psi0)):
                psi0[ind][4] /= np.sqrt(norm)
                sum += np.conj(psi0[ind][4]) * psi0[ind][4]
            print(psi0)


        elif params['ini_state'] == "projection":
            print("generating initial wavefunction by projection of a given function onto our basis \n")

            if test == True:
                print("generating test wavefunction on a grid")
      
                """ plot the radial part """
                h_radial = lambda r,n,l: (2*r/n)**l * np.exp(-r/n) * genlaguerre(n-l-1,2*l+1)(2*r/n)
                radgrid = np.linspace(0.0,10.0,10)
                thetagrid = np.linspace(0,np.pi,10)
                phigrid  = np.linspace(0,2*np.pi,10)
                grid3d = np.meshgrid(radgrid,thetagrid,phigrid)
                nodes = list(zip(*(dim.flat for dim in grid3d)))
                print(np.shape(nodes))
                
                nodes = np.asarray(nodes)

                #for node in nodes:
                #    print(node)
                plt.plot(nodes[:,0],h_radial(nodes[:,0],3,0)**2 * nodes[:,0]**2)
                #plt.show()

                """plot the angular part"""
                theta_2d, phi_2d = np.meshgrid(thetagrid, phigrid)
                xyz_2d = np.array([np.sin(theta_2d) * np.sin(phi_2d), np.sin(theta_2d) * np.cos(phi_2d), np.cos(theta_2d)]) #2D grid of cartesian coordinates
                colormap = cm.ScalarMappable( cmap=plt.get_cmap("cool") )
                colormap.set_clim(-.45, .45)
                limit = .5

                plt.figure()
                ax = plt.gca(projection = "3d")
                
                Y_lm = sph_harm(1,1, phi_2d,theta_2d)
                print(np.shape(Y_lm))
                #r = np.abs(Y_lm.real)*xyz_2d #calculate a point in 3D cartesian space for each value of spherical harmonic at (theta,phi)
                #r = np.abs(iniwf(2.0,theta_2d, phi_2d))*xyz_2d #calculate a point in 3D cartesian space for each value of spherical harmonic at (theta,phi)  
                r = self.custom_wf(1.0,theta_2d, phi_2d)*xyz_2d 
                ax.plot_surface(r[0], r[1], r[2], facecolors=colormap.to_rgba(Y_lm.real), rstride=1, cstride=1)
                ax.set_xlim(-limit,limit)
                ax.set_ylim(-limit,limit)
                ax.set_zlim(-limit,limit)
                #plt.show()

                #save function in wf0grid.txt

                fl_out = open( params['ini_state_file_grid'],'w')
                for i in range(len(nodes)):
                    fl_out.write("%6.3f"%float(nodes[i][0])+"  %6.3f"%float(nodes[i][1])+"  %6.3f"%float(nodes[i][2])+" %6.3f"%self.custom_wf(nodes[i][0],nodes[i][1], nodes[i][2])+"\n")

            # read coefficients from file

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

            """ project the wavefunction onto our basis """

            """ 
            input: complex wavefunction on r,theta,phi grid
            return: set of coefficients in l,m,i,n basis""" 

            mymap = mapping(int(params['lmin']), int(params['lmax']), int(params['nbins']), int(params['nlobatto']))
            maparray, Nbas = mymap.gen_map()

            for elem in maparray:
                psi0.append([ elem[0], elem[1], self.overlap_grid(iniwf,elem[1],elem[0])])

        return psi0

    def overlap_grid(self,psi0,m,l):
        Nlag = 10
        x,w = laggauss(Nlag)
        #psi0 is the projected function on the grid
        myscheme = quadpy.u3.schemes["lebedev_025"]()

        overlap = 0.0
        for k in range(Nlag):
            overlap += w[k] *  myscheme.integrate_spherical(lambda theta_phi: np.conjugate(sph_harm(m, l,  theta_phi[1]+np.pi, theta_phi[0])) * sph_harm(m, l,  theta_phi[1]+np.pi, theta_phi[0])  )

        #print(myscheme)
        """
        Symbols: 
                theta_phi[0] = theta in [0,pi]
                theta_phi[1] = phi  in [-pi,pi]
                sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) means that we put the phi angle in range [0,2pi] and the  theta angle in range [0,pi] as required by the scipy special funciton sph_harm
        """
        return overlap

    def custom_wf(self,r,theta,phi):
        """ return custom function in the hydrogen atom orbital basis"""
        h_radial = lambda r,n,l: (2*r/n)**l * np.exp(-r/n) * genlaguerre(n-l-1,2*l+1)(2*r/n)
        nmax = 2
        mmax = 1 
        lmax = 1
        f = 0.0
        counter =0
        """wavefunction definition"""
        for n in range(1,nmax+1):
            for l in range(0,n):
                counter += 1
                for m in range(-l,l+1):
                    print(n,l,m)

                    f +=   h_radial(r,n,l) * sph_harm(m,l, phi,theta)
        f /= np.sqrt(float(counter)) 
        return f.real





    def plot_wf_2d(self):
        pass

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
    """==== propagation parameters ===="""
    params['t0'] = 0.0
    params['tend'] = 0.0
    params['dt'] = 0.01

    
    """====basis set parameters===="""


    params['nlobatto'] = 4
    params['nbins'] = 1 #bug when nbins > nlobatto  
    params['binwidth'] = 4.0
    params['rshift'] = 1e-3 #rshift must be chosen such that it is non-zero and does not cover significant probability density region of any eigenfunction.

    params['lmin'] = 0
    params['lmax'] = 2
    
    """====runtime controls===="""
    params['method'] = "static"
    params['ini_state'] = "projection" #or "file" or "projection" or "manual"
    params['ini_state_file_coeffs'] = "wf0coeffs.txt" # name of file with coefficients of the initial wavefunction in our basis
    params['ini_state_file_grid'] = "wf0grid.txt" #initial wavefunction on a 3D grid of (r,theta,phi)
    params['basis'] = "prim" # or adiab
    params['potential'] = "hydrogen"
    params['scheme'] = "lebedev_025"


    """====field controls===="""
    params['field_type'] = "analytic" #or file
    params['field'] = "static_uniform"
    params['E0'] = 0.1

    """ We have two options for calculating the potential matrix elements: 
    1) cartesian: we use cartesian elements of the dipole moment operator and the electric field. Lebedev quadratures are used to compute matrix elements
    2) spherical-tensor: sphereical tensor representation of the field and the dipole moment. The matrix elements are computed analytically with 3-j symbols.
    """
   
    params['int_rep_type'] = 'cartesian'

    hydrogen = propagate()
    print(hydrogen.ini_wf(params,test=True))
    #hmat = hydrogen.prop_wf(params)
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