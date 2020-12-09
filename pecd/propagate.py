#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
import sys
import matelem


class propagate():

    def __init__(self):
        pass

    def prop_wf(self,method,basis,ini_state,params,potential,field,scheme):
        """ main method for propagating the wavefunction"""
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
            params (dict):      dictionary with all relevant numerical parameters of the calculation: t0,tend,dt,lmin,lmax,nbins,nlobatto,binwidth,tolerances
            potential (str):    name of the potential energy function (for now it is in an analytic form)
            field (str):        name of the electric field function used (it can further read from file or return analytic field)
            scheme (str):       quadrature scheme used in angular integration
        """

        if method == 'static':
            hamiltonian = matelem.hmat(params,potential,field,scheme,t=params['t0'])
            hmat = hamiltonian.calc_hmat()


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
    params['dt'] = 1.0

    
    """====basis set parameters===="""
    
    params['nlobatto'] = 4
    params['nbins'] = 2
    params['binwidth'] = 1.0
    params['rshift'] = 0.01

    params['lmin'] = 0
    params['lmax'] = 1
    
    """====runtime controls===="""

    method = "static"

    ini_state = "manual"

    basis = "prim"

    potential = "hydrogen"

    field = "field.txt"

    scheme = "lebedev"


    hydrogen = propagate()

    hydrogen.prop_wf(method,basis,ini_state,params,potential,field,scheme)