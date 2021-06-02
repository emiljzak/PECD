#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
from scipy import sparse
from scipy.fftpack import fftn
from scipy.sparse.linalg import expm, expm_multiply, eigsh
from scipy.special import sph_harm
from scipy.special import eval_legendre

from sympy.physics.wigner import gaunt
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.quantum.cg import CG
from sympy import N

import itertools

from pyhank import qdht, iqdht, HankelTransform

import MAPPING
import input
import GRID
import BOUND
import CONSTANTS
import FIELD
import PLOTS
#import ROTDENS

import time
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker

#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput
#from pycallgraph import Config



def prop_wf( params, ham0, psi_init, maparray, Gr, euler, ieuler ):

    time_to_au = CONSTANTS.time_to_au[ params['time_units'] ]

 
    #rho =  sparse.csc_matrix(ham_init).getnnz() / np.prod(sparse.csc_matrix(ham_init).shape)
    #print("density of the sparse hamiltonian matrix = " + str(rho) )

    #rho =  ham1.getnnz() / np.prod(ham1.shape)
    #print("density of the sparse hamiltonian matrix after filter = " + str(rho) )
    #exit()
    #plt.spy(ham1,precision=params['sph_quad_tol'], markersize=2)
    #plt.show()
    

    #BOUND.plot_mat(ham_init.todense())
    #plt.spy(ham_init, precision=params['sph_quad_tol'], markersize=5)
    #plt.show()
    #ham0 /= 2.0


    Nbas = len(psi_init)
    print("Nbas = " + str(Nbas))

    print("Setting up time-grid")
    tgrid = np.linspace(    params['t0'] * time_to_au, 
                            params['tmax'] * time_to_au, 
                            int((params['tmax']-params['t0'])/params['dt']+1), 
                            endpoint = True )
    dt = params['dt'] * time_to_au

    print("Allocating wavepacket")
    flwavepacket      = open( params['working_dir'] + params['wavepacket_file'] + "_" + str(ieuler) + ".dat", 'w' )
    wavepacket        = np.zeros( ( len(tgrid), Nbas ) , dtype=complex )
    #psi               = np.zeros( Nbas, dtype=complex ) 
    psi               = psi_init
    psi[:]           /= np.sqrt( np.sum( np.conj(psi) * psi ) )


    print(" Initialize the interaction matrix ")
    intmat0 = []# np.zeros(( Nbas , Nbas, 3 ), dtype = complex)
    intmat0.append(calc_intmat( [1.0, 0.0, 0.0], maparray, Gr, Nbas) ) #-1
    intmat0.append(calc_intmat( [0.0, 1.0, 0.0], maparray, Gr, Nbas)  )#0
    intmat0.append(calc_intmat( [0.0, 0.0, 1.0], maparray, Gr, Nbas)  ) #+1


    print("initialize electric field")
    Elfield = FIELD.Field(params)
    Fvec = Elfield.gen_field(tgrid) 
    
    if params['plot_elfield'] == True:
        PLOTS.plot_elfield(Fvec,tgrid,time_to_au)

    Fvec = np.asarray(Fvec)
    Fvec = np.stack(( Fvec[i] for i in range(len(Fvec)) ), axis=1) 
    Fvec += np.conjugate(Fvec)

    start_time_global = time.time()
    for itime, t in enumerate(tgrid): 

        start_time = time.time()
        print("t = " + str( "%10.1f"%(t/time_to_au)) + " as" + " normalization: " + str(np.sqrt( np.sum( np.conj(psi) * psi )) ) ) 
    
        #dip =   np.tensordot( Fvec[itime], intmat0, axes=([0],[2]) ) 
        #dip =   Elfield.gen_field(t)[0] * intmat0[:,:,0]  + Elfield.gen_field(t)[2] * intmat0[:,:,2]
        dip = Fvec[itime][0] * intmat0[0]  + Fvec[itime][2] * intmat0[2]

        #dip = sparse.csr_matrix(dip)

        #print("Is the full hamiltonian matrix symmetric? " + str(check_symmetric( ham0 + dip )))
                
        psi_out             = expm_multiply( -1.0j * ( ham0 + dip ) * dt, psi ) 
        wavepacket[itime,:] = psi_out
        psi                 = wavepacket[itime,:]

        flwavepacket.write( '{:10.3f}'.format(t) + 
                            " ".join('{:16.8e}'.format(psi[i].real) + '{:16.8e}'.format(psi[i].imag) for i in range(0,Nbas)) +\
                            '{:15.8f}'.format(np.sqrt(np.sum((psi[:].real)**2+(psi[:].imag)**2))) + "\n")
        
        end_time = time.time()
        print("time =  " + str("%10.3f"%(end_time-start_time)) + "s")

    end_time_global = time.time()
    print("The time for the wavefunction propagation is: " + str("%10.3f"%(end_time_global-start_time_global)) + "s")

    print("=====================================")
    print("==post-processing of the wavepacket==")
    print("====================================="+"\n")

    """ for post-processing choose analyze mode """

    print("==================================")
    print("== Momentum space wavefunctions ==")
    print("==================================")


    print("=========")
    print("==Plots==")
    print("=========")
        
    if params['plot_modes']['snapshot'] == True:
        plot_times = calc_plot_times(params,tgrid,dt)
        #re-check this!
        maparray = np.asarray(maparray)
        nbins = params['bound_nbins'] + params['nbins']
        
        Gr_all, Nr_all = GRID.r_grid_prim( params['bound_nlobs'], nbins , params['bound_binw'],  params['bound_rshift'] )

        maparray_chi, Nbas_chi = MAPPING.GENMAP_FEMLIST( params['FEMLIST'],  0, \
                                     params['map_type'], params['working_dir'] )

        flist = PLOTS.interpolate_chi(Gr_all, params['bound_nlobs'], nbins, params['bound_binw'], maparray_chi)

        for itime, t in enumerate(tgrid): 
            for ielem in plot_times:
                if itime == ielem:
                    print("Generating plot at time = " + str(t))
                    psi[:] = wavepacket[itime,:] 
                    PLOTS.plot_snapshot_int(params, psi, maparray, Gr_all, t, flist)



def BUILD_HMAT(params, Gr, maparray, Nbas):

    if params['read_ham_init_file'] == True:

        if params['hmat_format']   == "numpy_arr":
            if os.path.isfile(params['working_dir'] + params['file_hmat_init'] ):
        
                print (params['file_hmat_init'] + " file exist")
                hmat = read_ham_init(params)
                """ diagonalize hmat """
                start_time = time.time()
                enr, coeffs = np.linalg.eigh(hmat, UPLO = 'U')
                end_time = time.time()
                print("Time for diagonalization of field-free Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")

                #BOUND.plot_wf_rad(  0.0, params['bound_binw']* ( params['bound_nbins'] + params['nbins']), 1000, \
                #                    coeffs, maparray, Gr, params['bound_nlobs'], \
                #                    params['bound_nbins'] + params['nbins'])
                #PLOTS.plot_chi( 0.0, params['bound_binw'] * params['bound_nbins'],
                #                1000, Gr, params['bound_nlobs'], params['bound_nbins'])


                #BOUND.plot_mat(hmat)
                #plt.spy(hmat,precision=params['sph_quad_tol'], markersize=2)
                #plt.show()
                return hmat, coeffs
            else:
                raise ValueError("Incorrect file name for the Hamiltonian matrix")
                exit()

        elif params['hmat_format']   == "sparse_csr":

            if os.path.isfile(params['working_dir'] + params['file_hmat_init']+".npz" ):
                print (params['file_hmat_init']+ ".npz" + " file exist")
                ham0 = read_ham_init(params)

                """ diagonalize hmat """
                start_time = time.time()
                enr, coeffs = eigsh(ham0, k = params['num_ini_vec'], which='SA', return_eigenvectors = True, mode='normal')
                end_time = time.time()
                print("Time for diagonalization of field-free Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")

                return ham0, coeffs
            else:
                raise ValueError("Incorrect file name for the Hamiltonian matrix")
                exit()
    else:

        if params['hmat_format'] == 'numpy_arr':    
            hmat =  np.zeros((Nbas, Nbas), dtype=float)
        elif params['hmat_format'] == 'sparse_csr':
            hmat = sparse.csr_matrix((Nbas, Nbas), dtype=float)
            #hmat =  np.zeros((Nbas, Nbas), dtype=float)
        else:
            raise ValueError("Incorrect format type for the Hamiltonian")
            exit()

        """ calculate POTMAT """
        potmat, potind = BOUND.BUILD_POTMAT0( params, maparray, Nbas, Gr )      
        for ielem, elem in enumerate(potmat):
            hmat[ potind[ielem][0], potind[ielem][1] ] = elem[0]

        """ calculate KEO """
        start_time = time.time()
        keomat = BOUND.BUILD_KEOMAT_FAST( params, maparray, Nbas , Gr )
        end_time = time.time()
        print("New implementation - time for construction of KEO matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")

        #start_time = time.time()
        #keomat = BOUND.BUILD_KEOMAT( params, maparray, Nbas , Gr )
        #end_time = time.time()
        #print("Old implementation - time for construction of KEO matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")
        
        hmat += keomat 

        #print("plot of hmat")
        #BOUND.plot_mat(hmat)
        #plt.spy(hmat,precision=params['sph_quad_tol'], markersize=3, label="HMAT")
        #plt.legend()
        #plt.show()
        
        """ --- make the hamiltonian matrix hermitian --- """
        if params['hmat_format'] == 'numpy_arr':    
            ham0 = np.copy(hmat)
            ham0 += np.transpose(hmat.conjugate()) 
            for i in range(ham0.shape[0]):
                ham0[i,i] -= hmat.diagonal()[i]
            print("Is the field-free hamiltonian matrix symmetric? " + str(check_symmetric(ham0)))

        elif params['hmat_format'] == 'sparse_csr':
            #hmat = sparse.csr_matrix(hmat)
            hmat_csr_size = hmat.data.size/(1024**2)
            print('Size of the sparse Hamiltonian csr_matrix: '+ '%3.2f' %hmat_csr_size + ' MB')
            ham0 = hmat.copy()
            ham0 += hmat.getH()
            for i in range(ham0.shape[0]):
                ham0[i,i] -= hmat.diagonal()[i]
        else:
            raise ValueError("Incorrect format type for the Hamiltonian")
            exit()

        """ --- filter hamiltonian matrix  --- """

        if params['hmat_format'] == 'numpy_arr':    
            ham_filtered = np.where( np.abs(ham0) < params['hmat_filter'], 0.0, ham0)
            #ham_filtered = sparse.csr_matrix(ham_filtered)

        elif params['hmat_format'] == 'sparse_csr':
            nonzero_mask = np.array(np.abs(ham0[ham0.nonzero()]) < params['hmat_filter'])[0]
            rows = ham0.nonzero()[0][nonzero_mask]
            cols = ham0.nonzero()[1][nonzero_mask]
            ham0[rows, cols] = 0
            ham_filtered = ham0.copy()


        """ diagonalize hmat """
        if params['hmat_format'] == 'numpy_arr':    
            start_time = time.time()
            enr, coeffs = np.linalg.eigh(ham_filtered , UPLO = 'U')
            end_time = time.time()
        elif params['hmat_format'] == 'sparse_csr':
            start_time = time.time()
            enr, coeffs = eigsh(ham_filtered, k = params['num_ini_vec'], which='SA', return_eigenvectors=True, mode='normal')
            end_time = time.time()

   
        print("Time for diagonalization of field-free Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")

        #BOUND.plot_wf_rad(  0.0, params['bound_binw']* ( params['bound_nbins'] + params['nbins']), 1000, \
        #                    coeffs, maparray, Gr, params['bound_nlobs'], \
        #                    params['bound_nbins'] + params['nbins'])
        #exit()
        
        #PLOTS.plot_chi( 0.0, params['bound_binw'] * params['bound_nbins'],
        #                1000, Gr, params['bound_nlobs'], params['bound_nbins'])

        print("Normalization of initial wavefunctions: ")
        for v in range(params['num_ini_vec']):
            print(str(v) + " " + str(np.sqrt( np.sum( np.conj(coeffs[:,v] ) * coeffs[:,v] ) )))

        if params['save_ham_init'] == True:
            if params['hmat_format'] == 'sparse_csr':
                sparse.save_npz( params['working_dir'] + params['file_hmat_init'] , ham0 , compressed = False )
            elif params['hmat_format'] == 'numpy_arr':
                with open( params['working_dir'] + params['file_hmat_init'] , 'w') as hmatfile:   
                    np.savetxt(hmatfile, ham0, fmt = '%10.4e')

        if params['save_psi_init'] == True:
            psifile = open(params['working_dir'] + params['file_psi_init'], 'w')
            for ielem,elem in enumerate(maparray):
                psifile.write( " %5d"%elem[0] +  " %5d"%elem[1] + "  %5d"%elem[2] + \
                                " %5d"%elem[3] +  " %5d"%elem[4] + "\t" + \
                                "\t\t ".join('{:10.5e}'.format(coeffs[ielem,v]) for v in range(0,params['num_ini_vec'])) + "\n")

        if params['save_enr_init'] == True:
            with open(params['working_dir'] + params['file_enr_init'], "w") as energyfile:   
                np.savetxt( energyfile, enr * CONSTANTS.au_to_ev , fmt='%10.5f' )
    

        return ham0, coeffs


def read_coeffs(filename,nvecs):

    coeffs = []
    fl = open( filename , 'r' )
    for line in fl:
        words   = line.split()
        i       = int(words[0])
        n       = int(words[1])
        xi      = int(words[2])
        l       = int(words[3])
        m       = int(words[4])
        c       = []
        for ivec in range(nvecs):
            c.append(float(words[5+ivec]))
        coeffs.append([i,n,xi,l,m,np.asarray(c)])
    return coeffs

def proj_wf0_wfinit_dvr(coeffs0, marray, Nbas_global):
    psi = []
    for ivec in range(Nbas_global):
        if ivec < len(coeffs0):
            psi.append([marray[ivec][0],marray[ivec][1],marray[ivec][2],\
                        marray[ivec][3],marray[ivec][4],coeffs0[ivec][5]])
        else:
            psi.append([marray[ivec][0],marray[ivec][1],marray[ivec][2],\
                        marray[ivec][3],marray[ivec][4],0.0])
    """ for ivec in range(Nbas_global):
        for icoeffs0 in coeffs:
            if  marray[ivec][0] == icoeffs0[0]   and \
                marray[ivec][1] == icoeffs0[1]   and \
                marray[ivec][3] == icoeffs0[3]   and \
                marray[ivec][4] == icoeffs0[4]: #i,n,l,m
                    psi.append([marray[ivec][0],marray[ivec][1],marray[ivec][2],\
                                marray[ivec][3],marray[ivec][4],icoeffs[5]])
                break

        psi.append([marray[ivec][0],marray[ivec][1],marray[ivec][2],\
                    marray[ivec][3],marray[ivec][4],icoeffs[5]])
    """
    return psi


def read_ham_init(params):
    if params['hmat_format'] == 'sparse_csr':
        hmat = sparse.load_npz( params['working_dir'] + params['file_hmat_init']+ ".npz" )
    elif params['hmat_format'] == 'numpy_arr':
        with open( params['working_dir'] + params['file_hmat_init'] , 'r') as hmatfile:   
            hmat = np.loadtxt(hmatfile)
    return hmat


def calc_intmat(field,maparray,rgrid,Nbas):  

    #field: (E_-1, E_0, E_1) in spherical tensor form
    """calculate the <Y_l'm'(theta,phi)| d(theta,phi) | Y_lm(theta,phi)> integral """

    if params['hmat_format'] == 'numpy_arr':    
        intmat =   np.zeros(( Nbas , Nbas ), dtype = complex)
    elif params['hmat_format'] == 'sparse_csr':
        intmat = sparse.csr_matrix(( Nbas, Nbas ), dtype = complex)
    


    D = np.zeros(3)
    for i in range(Nbas):
        rin = rgrid[ maparray[i][0], maparray[i][1] -1 ]
        for j in range(Nbas):
            if  maparray[i][2] == maparray[j][2]:
                D[0] = N( gaunt( maparray[i][3], 1, maparray[j][3], maparray[i][4], -1, maparray[j][4] ) )
                D[1] = N( gaunt( maparray[i][3], 1, maparray[j][3], maparray[i][4], 0, maparray[j][4] ) ) * np.sqrt(2.)
                D[2] = N( gaunt( maparray[i][3], 1, maparray[j][3], maparray[i][4], 1, maparray[j][4] ) ) 

                intmat[i,j] = np.sqrt( 2.0 * np.pi / 3.0 ) * np.dot(field, D) * rin 

    #plt.spy(intmat, precision=params['sph_quad_tol'], markersize=5)
    #plt.show()

    #intmat += np.conjugate(intmat.T)

    #print("Interaction matrix")
    #with np.printoptions(precision=4, suppress=True, formatter={'complex': '{:15.8f}'.format}, linewidth=400):
    #    print(intmat)
    #print("Is the interaction matrix symmetric? " + str(check_symmetric(intmat)))

    return intmat


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.conj(), rtol=rtol, atol=atol)


def calc_plot_times(params,tgrid,dt):
    time_to_au = CONSTANTS.time_to_au[ params['time_units'] ]
    plot_times = []
    for index,item in enumerate(params['plot_controls']["plottimes"]):
        if int( item * time_to_au / dt ) > len(tgrid):
            print("removing time: " + str(item) + " from plotting list. Time exceeds the propagation time-grid!")
        else:
            plot_times.append( int(item * time_to_au / dt) )
    print("Final list of plottime indices in tgrid:")
    print(plot_times)
    return plot_times


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

def cart2sph(x,y,z):
    r=np.sqrt(x**2+y**2+z**2)
    theta=np.arctan(np.sqrt(x**2+y**2)/z)
    phi=np.arctan(y/x)
    return r,theta,phi
    
def spharmcart(l,m,x,y,z):
    tol = 1e-8
    if all(np.abs(x)>tol) and all(np.abs(z)>tol): 
        r       =   np.sqrt(x**2+y**2+z**2)
        theta   =   np.arctan(np.sqrt(x**2+y**2)/z)
        phi     =   np.arctan(y/x)
    else:
        r       = 0.0
        theta   = 0.0
        phi     = 0.0
    return sph_harm(m, l, phi, theta)

def calc_fftcart_psi_3d(params, maparray, Gr, psi, chilist):
    coeff_thr = 1e-3
    ncontours = 20

    nlobs   = params['nlobs']
    nbins   = params['bound_nbins'] + params['nbins'] 
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


def calc_partial_waves(chilist, grid_r, lmax, psi, maparray_global, maparray_chi):
    """
    returns: list of numpy arrays. List is labelled by l,m.
    """
    Plm = []

    Nbas = len(maparray_global)
    Nr = len(maparray_chi)
    print("number of radial basis points/functions: " + str(len(maparray_chi)))

    npts = grid_r.size

    #for i in range(359):
    #    plt.plot(grid_r,chilist[i](grid_r))
    #plt.show()


    val = np.zeros(npts, dtype = complex)

    coeffs = np.zeros(Nbas, dtype = complex)
    for ielem in range(Nbas):
        coeffs[ielem] =  psi[2*ielem] + 1j * psi[2*ielem + 1] 

    c_arr = coeffs.reshape(len(maparray_chi),-1)

    indang = 0
    for l in range(0,lmax+1):
        for m in range(-l,l+1):
            print(l,m)
        
            for ielem, elem in enumerate(maparray_chi):
                if elem[0] > -1: #cut-out bound-state electron density

                    val +=  c_arr[ielem][indang] *  chilist[elem[2]-1](grid_r)

            indang += 1
            Plm.append([l,m,val])
            val = 0.0 + 1j * 0.0
    #for s in range(16):
    #    plt.scatter(grid_r,np.abs(Plm[s][2]),marker='.',label="P_"+str(s))
    #    plt.legend()
    #plt.show()
 
    return Plm


def calc_hankel_transforms(Plm, grid_r):
    Flm = [] #list of output Hankel transforms
  
    for ielem, elem in enumerate(Plm):
        print("Calculating Hankel transform for partial wave Plm: " + str(elem[0]) + " " + str(elem[1]))

        Hank_obj = HankelTransform(elem[0], radial_grid = grid_r) #max_radius=200.0, n_points=1000) #radial_grid=fine_grid)
        #Hank.append(Hank_obj) 
        Plm_resampled = Hank_obj.to_transform_r(elem[2])
        F = Hank_obj.qdht(Plm_resampled)
        Flm.append([elem[0],elem[1],F])

        #plt.plot(Hank_obj.kr,np.abs(F))
    #plt.show()
    return Flm, Hank_obj.kr

def calc_FT_3D_hankel(Plm, Flm, kgrid, lmax, grid_theta, grid_r, maparray_chi, maparray_global, psi, chilist, phi0 = 0.0 ):
    """ returns: fourier transform inside a ball grid (r,theta,phi) """

    npts      = grid_r.size

    FT = np.zeros((npts  ,npts  ), dtype = complex)

    for i in range(npts ):
        #print(i)
        for elem in Flm:
            FT[i,:] +=   ((-1.0 * 1j)**elem[0]) * elem[2][:npts] * PLOTS.spharm(elem[0], elem[1], grid_theta[i] , phi0) 

    return FT, kgrid

def plot_W_3D_num(params, maparray_chi, maparray_global, psi, chilist, phi0 = 0.0):
    ncontours = 100
    grid_theta, grid_r = calc_grid_for_FT(params)

    #calculate partial waves on radial grid
    Plm = calc_partial_waves(chilist, grid_r, params['bound_lmax'], psi, maparray_global, maparray_chi)

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



def precompute_spharm(grid_theta,lmax, phi0):
    SP = {}

    for l in range(0,2*lmax+1):
        for m in range(-l,l+1):
            SP[ str(l) + ',' + str(m) ] = PLOTS.spharm(l, m, grid_theta , phi0) 


    return SP



def calc_W_3D_analytic(lmax, grid_theta, grid_r, maparray_chi, maparray_global, psi, chilist, phi0 = 0.0 ):
    """ returns: square modulus of fourier transform inside a ball grid (r,theta,phi) calculated with an analytic expression"""
    npts      = grid_r.size
    #Precompute spherical harmonics on the theta grid
    SP = precompute_spharm(grid_theta,lmax, phi0)

    #plt.plot(grid_theta,PLOTS.spharm(2, 1, grid_theta, phi0))
    #plt.show()

    #calculate partial waves on radial grid
    Plm = calc_partial_waves(chilist, grid_r, lmax, psi, maparray_global, maparray_chi)

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


def plot_W_3D_analytic(params, maparray_chi, maparray_global, psi, chilist, phi0 = 0.0):
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




def calc_W_2D_av_phi_num(params, maparray_chi, maparray_global, psi, chilist):
    """calculate numerically average of W over the phi angle"""

    Nphi = 36
    phigrid = np.linspace(0, 2.0 * np.pi, Nphi, endpoint=False)
    grid_theta, grid_r = calc_grid_for_FT(params)
   
    W_arr = []

    #calculate partial waves on radial grid
    Plm = calc_partial_waves(chilist, grid_r, params['bound_lmax'], psi, maparray_global, maparray_chi)

    #calculate Hankel transforms on appropriate k-vector grid
    Flm, kgrid = calc_hankel_transforms(Plm, grid_r)


    for iphi in range(Nphi):
        FT, kgrid = calc_FT_3D_hankel(Plm, Flm, kgrid, params['bound_lmax'], grid_theta, grid_r, maparray_chi, maparray_global, psi, chilist, phigrid[iphi] )
        W_arr.append( np.abs(FT)**2 ) 


    Wav = np.copy(W_arr[0])
    Wav = 0.0

    for elem in W_arr:
        Wav += elem

    return Wav / float(Nphi), kgrid

def plot_W_2D_av_phi_num(params, maparray_chi, maparray_global, psi, chilist):
    ncontours = 100
    grid_theta, grid_r = calc_grid_for_FT(params)

    Wav, kgrid  = calc_W_2D_av_phi_num(params, maparray_chi, maparray_global, psi, chilist)

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



def calc_W_av_phi_analytic(params, maparray_chi, maparray_global, psi, chilist):
    """ returns: square modulus of fourier transform inside a disk (r,theta) calculated with analytic expression
        and averaged over phi"""

    grid_theta, grid_r = calc_grid_for_FT(params)
    npts      = grid_r.size

    #calculate partial waves on radial grid
    Plm = calc_partial_waves(chilist, grid_r, params['bound_lmax'], psi, maparray_global, maparray_chi)

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

def plot_W_2D_av_phi_analytic(params, maparray_chi, maparray_global, psi, chilist):
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


def calc_ftpsi_2d(params, maparray, Gr, psi, chilist):
   
    coeff_thr = 1e-5
    ncontours = 100

    nlobs   = params['nlobs']
    nbins   = params['bound_nbins'] + params['nbins'] 
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

def calc_pecd(file_lcpl,file_rcpl, params, maparray_global, Gr, chilist):
    
    Nbas = len(maparray_global)

    itime = int( params['time_pecd'] / params['dt']) 


    psi_rcpl =  read_wavepacket(file_rcpl, itime, Nbas)
    psi_lcpl =  read_wavepacket(file_rcpl, itime, Nbas)


    ft_rcpl, yftgrid, zftgrid = calc_ftpsi_2d(params, maparray_global, Gr, psi_rcpl, chilist)
    ft_lcpl, yftgrid, zftgrid = calc_ftpsi_2d(params, maparray_global, Gr, psi_lcpl, chilist)


    ncontours = 100

    nlobs   = params['nlobs']
    nbins   = params['bound_nbins'] + params['nbins'] 
    npoints = 200
    rmax    = nbins * params['bound_binw']
    rmin    = 10.0

    fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    axpecd = fig.add_subplot(spec[0, 0])

    line_pecd = axpecd.contourf(yftgrid, zftgrid , ( ft_rcpl[:npoints].imag - ft_lcpl[:npoints].imag ) , 
                                        ncontours, cmap = 'jet') #vmin=0.0, vmax=1.0cmap = jet, gnuplot, gnuplot2
    plt.colorbar(line_pecd, ax=axpecd, aspect=30)
    plt.legend()   
    plt.show()  

def calc_grid_for_FT(params):
    """ Calculate real-space grid (r,theta) for evaluation of Hankel transform and for plottting"""
    """ The real-space grid determines the k-space grid returned by PyHank """

    nbins   = params['bound_nbins'] + params['nbins'] 
    rmax    = nbins * params['bound_binw']
    npts    = params['N_r_points']
    N_red   = npts 

    grid_theta  = np.linspace(-np.pi, np.pi, N_red , endpoint = False ) # 
    grid_r      = np.linspace(0.0, rmax, npts, endpoint = False)

    return grid_theta, grid_r


def gen_euler_grid(n_euler):
    alpha_1d        = list(np.linspace(0, 2*np.pi,  num=n_euler, endpoint=False))
    beta_1d         = list(np.linspace(0, np.pi,    num=n_euler, endpoint=True))
    gamma_1d        = list(np.linspace(0, 2*np.pi,  num=n_euler, endpoint=False))
    euler_grid_3d   = np.array(list(itertools.product(*[alpha_1d, beta_1d, gamma_1d]))) #cartesian product of [alpha,beta,gamma]

    #we can choose meshgrid instead
    #euler_grid_3d_mesh = np.meshgrid(alpha_1d, beta_1d, gamma_1d)
    #print(euler_grid_3d_mesh[0].shape)
    #print(np.vstack(euler_grid_3d_mesh).reshape(3,-1).T)
    #print(euler_grid_3d)
    #exit()
    n_euler_3d      = euler_grid_3d.shape[0]
    print("\nTotal number of 3D-Euler grid points: ", n_euler_3d , " and the shape of the 3D grid array is:    ", euler_grid_3d.shape)
    #print(euler_grid_3d)
    return euler_grid_3d, n_euler_3d


if __name__ == "__main__":      

	params = input.gen_input()

	maparray_global, Nbas_global = MAPPING.GENMAP_FEMLIST(  params['FEMLIST'],
                                                            params['bound_lmax'],
                                                            params['map_type'],
                                                            params['working_dir'] )

	Gr, Nr                       = GRID.r_grid(             params['bound_nlobs'], 
                                                            params['bound_nbins'] + params['nbins'], 
                                                            params['bound_binw'],  
                                                            params['bound_rshift'] )


	if params['mode'] == 'propagate_single':

		ham_init, psi_init = BUILD_HMAT(params, Gr, maparray_global, Nbas_global)
        
        #graphviz = GraphvizOutput(output_file=params['working_dir']+'BUILD_HMAT.png')
        #config = Config(max_depth=4)
        #with PyCallGraph(output=graphviz, config=config):
        #print(ham0)
        #plt.spy(ham_init, precision=params['sph_quad_tol'], markersize=5)
        #plt.show()

		prop_wf(params, ham_init, psi_init[:,params['ivec']], maparray_global, Gr, params['euler0'], 0)

	elif params['mode'] == 'analyze_single':

		itime = int( params['analyze_time'] / params['dt']) 

		if params['analyze_mpad'] == True:
			#read wavepacket from file
			file_wavepacket      = params['working_dir'] + params['wavepacket_file'] + ".dat"
			psi =  read_wavepacket(file_wavepacket, itime, Nbas_global)

			#print(np.shape(psi))
			nbins = params['bound_nbins'] + params['nbins']
			
			Gr_prim, Nr_prim = GRID.r_grid_prim( params['bound_nlobs'], nbins , params['bound_binw'],  params['bound_rshift'] )

			maparray_chi, Nbas_chi = MAPPING.GENMAP_FEMLIST( params['FEMLIST'],  0, \
				                        params['map_type'], params['working_dir'] )

			chilist = PLOTS.interpolate_chi(Gr_prim, params['bound_nlobs'], nbins, params['bound_binw'], maparray_chi)

			#calc_ftpsi_2d(params, maparray_global, Gr, psi, chilist)

			if params['FT_method']  == "FFT_cart":
				calc_fftcart_psi_3d(params, maparray_global, Gr, psi, chilist)

			elif params['FT_method']  == "FFT_hankel":
				
				# PLOTS of W:
				#plot_W_3D_num(params, maparray_chi, maparray_global, psi, chilist, 0.0)
				#plot_W_3D_analytic(params, maparray_chi, maparray_global, psi, chilist, 0.0)
				plot_W_2D_av_phi_num(params, maparray_chi, maparray_global, psi, chilist)
				#plot_W_2D_av_phi_analytic(params, maparray_chi, maparray_global, psi, chilist)
				#grid_theta, grid_r = calc_grid_for_FT(params)
				#calc_fthankel_psi_3d( params['bound_lmax'], grid_theta, grid_r , maparray_chi, maparray_global, psi, chilist, phi=0.0)
				#calc_W_analytic(params, maparray_chi, maparray_global, psi, chilist)
				#calc_W_av_phi_analytic(params, maparray_chi, maparray_global, psi, chilist)

		if params['analyze_pecd'] == True:
			file_rcpl = params['working_dir'] + "wavepacket_RCPL.dat"
			file_lcpl = params['working_dir'] + "wavepacket_LCPL.dat"

			calc_pecd(file_lcpl,file_rcpl, params, maparray_global, Gr, chilist)

	elif params['mode'] == 'propagate_grid':
		ibatch  = int(sys.argv[1])
		N_batch = int(sys.argv[2])
		N_Euler = int(sys.argv[3])
		print(ibatch, N_batch, N_Euler)
		N_per_batch = int(N_Euler**3/N_batch)
		print("number of points per batch = " + str(N_per_batch))
		grid_euler, n_grid_euler = gen_euler_grid(N_Euler)
		ham_init, psi_init = BUILD_HMAT(params, Gr, maparray_global, Nbas_global)

		for irun in range(ibatch * N_per_batch, (ibatch+1) * N_per_batch):
			print(grid_euler[irun])
			print(irun)
			prop_wf(params, ham_init, psi_init[:,params['ivec']], maparray_global, Gr, grid_euler[irun], irun)

	elif params['mode'] == 'analyze_grid':
        
		itime = int( params['analyze_time'] / params['dt']) 

		if params['analyze_mpad'] == True:
            N_Euler = int(sys.argv[3])

			grid_euler, n_grid_euler = gen_euler_grid(N_Euler)

			nbins = params['bound_nbins'] + params['nbins']
			
			Gr_prim, Nr_prim = GRID.r_grid_prim( params['bound_nlobs'], nbins , params['bound_binw'],  params['bound_rshift'] )

			maparray_chi, Nbas_chi = MAPPING.GENMAP_FEMLIST( params['FEMLIST'],  0, \
				                        params['map_type'], params['working_dir'] )

			chilist = PLOTS.interpolate_chi(Gr_prim, params['bound_nlobs'], nbins, params['bound_binw'], maparray_chi)

			grid_theta, grid_r = calc_grid_for_FT(params)
			
			Wav = np.zeros((grid_theta.shape[0],grid_r.shape[0]), dtype = float)
            
            ibatch  = int(sys.argv[1])
            N_batch = int(sys.argv[2])

            print(ibatch, N_batch, N_Euler)
            N_per_batch = int(N_Euler**3/N_batch)
            print("number of points per batch = " + str(N_per_batch))  
            
            for irun in range(ibatch * N_per_batch, (ibatch+1) * N_per_batch):
			    print(grid_euler[irun])
			    print(irun)
				alpha   = grid_euler[irun][0]
				beta    = grid_euler[irun][1]
				gamma   = grid_euler[irun][2]
				#read wavepacket from file
				file_wavepacket      = params['working_dir'] + params['wavepacket_file'] + "_" +str(irun) + ".dat"
				psi =  read_wavepacket(file_wavepacket, itime, Nbas_global)

				if params['FT_method']  == "FFT_cart":
				    calc_fftcart_psi_3d(params, maparray_global, Gr, psi, chilist)
				elif params['FT_method']  == "FFT_hankel":

				    #calculate partial waves on radial grid
				    Plm = calc_partial_waves(chilist, grid_r, params['bound_lmax'], psi, maparray_global, maparray_chi)

				    #calculate Hankel transforms on appropriate k-vector grid
				    Flm, kgrid = calc_hankel_transforms(Plm, grid_r)

				    FT, kgrid = calc_FT_3D_hankel(Plm, Flm, kgrid, params['bound_lmax'], grid_theta, grid_r, maparray_chi, maparray_global, psi, chilist, gamma )
				    
				    #rho = ROTDENS.calc_rotdens(grid_euler[ipoint]) #rotational density
				    #plot_W_3D_num(params, maparray_chi, maparray_global, psi, chilist, gamma)
				    Wav += np.abs(FT)**2
				    #PLOTS.plot_2D_polar_map(np.abs(FT)**2,grid_theta,kgrid,100)
			print(Wav)

            with open( params['working_dir'] + "W_av_3D_" + str(ibatch) , 'w') as Wavfile:   
                np.savetxt(Wavfile, Wav, fmt = '%10.4e')
			#PLOTS.plot_2D_polar_map(Wav,grid_theta,kgrid,100)


	else:
		raise ValueError("Incorrect execution mode keyword")
		exit()
