#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
import sys
import matelem
import timeit
from basis import radbas
from matelem import mapping
import matplotlib.pyplot as plt

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
            evals, coeffs = hamiltonian.calc_hmat()
            #print(type(coeffs))
            #print(np.shape(coeffs))
            #rbas.plot_wf_rad(0.0,params['nbins'] * params['binwidth'],1000,coeffs,maparray,rgrid)

    def plot_mat(self,mat):
        """ plot 2D array with color-coded magnitude"""
        pass

    def ini_wf(self,ini_state):
        """ plot 2D array with color-coded magnitude"""
        wf0=0
        return wf0

   

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


    params['nlobatto'] = 20
    params['nbins'] = 1
    params['binwidth'] = 40.0
    params['rshift'] = 1e-3 #rshift must be chosen such that it is non-zero and does not cover significant probability density region of any eigenfunction.

    params['lmin'] = 0
    params['lmax'] = 1
    
    """====runtime controls===="""
    params['method'] = "static"
    params['ini_state'] = "manual" #or file
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

    hydrogen.prop_wf(params)

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