#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#

import input
import MAPPING
import POTENTIAL
import GRID 

import numpy as np
from scipy import sparse
from scipy.special import sph_harm
import quadpy

import sys
import time

from numba import jit, prange

import matplotlib.pyplot as plt

""" start of @jit section """
@jit(nopython=True,parallel=False,cache = True, fastmath=False) 
def P(l, m, x):
	pmm = np.ones(1,)
	if m>0:
		somx2 = np.sqrt((1.0-x)*(1.0+x))
		fact = 1.0
		for i in range(1,m+1):
			pmm = pmm * (-1.0 * fact) * somx2
			fact = fact + 2.0
	
	if l==m :
		return pmm * np.ones(x.shape,)
	
	pmmp1 = x * (2.0*m+1.0) * pmm
	
	if l==m+1:
		return pmmp1
	
	pll = np.zeros(x.shape)
	for ll in range(m+2, l+1):
		pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m)
		pmm = pmmp1
		pmmp1 = pll
	
	return pll

#@jit(nopython=True,parallel=False,fastmath=False) 
def divfact(a, b):
	# PBRT style
	if (b == 0):
		return 1.0
	fa = a
	fb = abs(b)
	v = 1.0

	x = fa-fb+1.0
	while x <= fa+fb:
		v *= x
		x+=1.0

	return 1.0 / v

LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')
    
@jit(nopython=True,parallel=False,cache = True, fastmath=False) 
def fast_factorial(n):
    return LOOKUP_TABLE[n]

def factorial(x):
	if(x == 0):
		return 1.0
	return x * factorial(x-1)

@jit(nopython=True,parallel=False,cache = True, fastmath=False) 
def K(l, m):
	return np.sqrt( ((2 * l + 1) * fast_factorial(l-m)) / (4*np.pi*fast_factorial(l+m)) )

@jit(nopython=True,parallel=False,cache = True, fastmath=False) 
def SH(l, m, theta, phi):
	if m==0 :
		return K(l,m)*P(l,m,np.cos(theta))*np.ones(phi.shape,)
	elif m>0 :
		return np.sqrt(2.0)*K(l,m)*np.cos(m*phi)*P(l,m,np.cos(theta))
	else:
		return np.sqrt(2.0)*K(l,-m)*np.sin(-m*phi)*P(l,-m,np.cos(theta))

@jit( nopython=True, parallel=False, cache = True, fastmath=False) 
def calc_potmat_jit( vlist, VG, Gs ):
    pot = []
    potind = []
    for p1 in prange(vlist.shape[0]):
        #print(vlist[p1,:])
        w = Gs[vlist[p1,0]][:,2]
        G = Gs[vlist[p1,0]] 
        V = VG[vlist[p1,0]]

        f = SH( vlist[p1,1] , vlist[p1,2]  , G[:,0], G[:,1] + np.pi ) * \
            SH( vlist[p1,3] , vlist[p1,4]  , G[:,0], G[:,1] + np.pi ) * \
            V[:]

        pot.append( [np.dot(w,f.T) * 4.0 * np.pi ] )
        potind.append( [ vlist[p1,5], vlist[p1,6] ] )
        #potmat[vlist[p1,5],vlist[p1,6]] = np.dot(w,f.T) * 4.0 * np.pi
    return pot, potind


""" end of @jit section """


def BUILD_HMAT0(params):

    maparray, Nbas = MAPPING.GENMAP( params['bound_nlobs'], params['bound_nbins'], params['bound_lmax'], \
                                     params['map_type'], params['working_dir'] )

    Gr, Nr = GRID.r_grid( params['bound_nlobs'], params['bound_nbins'], params['bound_binw'],  params['bound_rshift'] )

    if params['hmat_format'] == 'csr':
        hmat = sparse.csr_matrix((Nbas, Nbas), dtype=np.float64)
    elif params['hmat_format'] == 'regular':
        hmat = np.zeros((Nbas, Nbas), dtype=np.float64)


    """ calculate hmat """

    potmat0, potind = BUILD_POTMAT0( params, maparray, Nbas , Gr )

    for ielem, elem in enumerate(potmat0):
        #print(potind[ielem][0],potind[ielem][1])
        hmat[ potind[ielem][0],potind[ielem][1] ] = elem[0]


    #plot_mat(hmat)
    plt.spy(hmat,precision=params['sph_quad_tol'], markersize=5)
    plt.show()
    
    """ diagonalize hmat """
    start_time = time.time()
    enr0, coeffs0 = np.linalg.eigh(hmat, UPLO = 'U')
    end_time = time.time()
    print("Time for diagonalization of field-free Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")

    print("Normalization of the wavefunction: ")
    for v in range(params['num_ini_vec']):
        print(str(v) + " " + str(np.sqrt( np.sum( np.conj(coeffs0[:,v] ) * coeffs0[:,v] ) )))

    if params['save_ham0'] == True:
        if params['hmat_format'] == 'csr':
            sparse.save_npz( params['working_dir'] + params['file_hmat0'] , hmat , compressed = False )
        elif params['hmat_format'] == 'regular':
            with open( params['working_dir'] + params['file_hmat0'] , 'w') as hmatfile:   
                np.savetxt(hmatfile, hmat, fmt = '%10.4e')

    if params['save_psi0'] == True:
        psi0file = open(params['working_dir'] + params['file_psi0'], 'w')
        for ielem,elem in enumerate(maparray):
            psi0file.write( " %5d"%elem[0] +  " %5d"%elem[1] + "  %5d"%elem[2] + \
                            " %5d"%elem[3] +  " %5d"%elem[4] + "\t" + \
                            "\t\t ".join('{:10.5e}'.format(coeffs0[ielem,v]) for v in range(0,params['num_ini_vec'])) + "\n")

    if params['save_enr0'] == True:
        with open(params['working_dir'] + params['file_enr0'], "w") as energyfile:   
            np.savetxt( energyfile, enr0 , fmt='%10.5f' )
  
    print(enr0)

def BUILD_KEOMAT0(params):
    print("under construction")


def BUILD_POTMAT0( params, maparray, Nbas , Gr ):


    print("Interpolating electrostatic potential")
    esp_interpolant = POTENTIAL.INTERP_POT(params)

    if  params['gen_adaptive_quads'] == True:
        sph_quad_list = gen_adaptive_quads( params, esp_interpolant, Gr )
    else:
        sph_quad_list = read_adaptive_quads(params)

    Gs = GRID.GEN_GRID( sph_quad_list )
    VG = POTENTIAL.BUILD_ESP_MAT( Gs, Gr, esp_interpolant, params['r_cutoff'] )
    vlist = MAPPING.GEN_VLIST( maparray, Nbas, params['map_type'] )
    vlist = np.asarray(vlist)

    if params['calc_method'] == 'jit':
        start_time = time.time()
        potmat0, potind = calc_potmat_jit( vlist, VG, Gs )
        end_time = time.time()
        print("First call: Time for construction of potential matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")

        #start_time = time.time()
        #potmat0, potind = calc_potmat_jit( vlist, VG, Gs )
        #end_time = time.time()
        #print("Second call: Time for construction of potential matrix is " +  str("%10.3f"%(end_time-start_time)) + "s")
    """
    if params['hmat_format'] == "regular":
        potmat = convert_lists_to_regular(potmat0,potind)
    elif params['hmat_format'] == "csr":
        potmat = convert_lists_to_csr(potmat0,potind)
    """
    return potmat0, potind

def calc_potmatelem_quadpy( l1, m1, l2, m2, rin, scheme, esp_interpolant ):
    """calculate single element of the potential matrix on an interpolated potential"""
    myscheme = quadpy.u3.schemes[scheme]()
    """
    Symbols: 
            theta_phi[0] = theta in [0,pi]
            theta_phi[1] = phi  in [-pi,pi]
            sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) means that we 
                                                                 put the phi angle in range [0,2pi] and 
                                                                 the  theta angle in range [0,pi] as required by 
                                                                 the scipy special funciton sph_harm
    """
    val = myscheme.integrate_spherical(lambda theta_phi: np.conjugate( sph_harm( m1, l1,  theta_phi[1]+np.pi, theta_phi[0] ) ) \
                                                                     * sph_harm( m2, l2, theta_phi[1]+np.pi, theta_phi[0] ) \
                                                                     * POTENTIAL.calc_interp_sph( esp_interpolant, rin, theta_phi[0], theta_phi[1]+np.pi) ) 

    return val

def read_adaptive_quads(params):
    levels = []
    quadfilename = params['working_dir'] + params['file_quad_levels'] 
    fl = open( quadfilename , 'r' )
    for line in fl:
        words   = line.split()
        i       = int(words[0])
        n       = int(words[1])
        xi      = int(words[2])
        scheme  = str(words[3])
        levels.append([i,n,xi,scheme])
    return levels

def gen_adaptive_quads(params, esp_interpolant, rgrid):

    sph_quad_list = [] #list of quadrature levels needed to achieve convergence quad_tol of the potential energy matrix elements (global)

    lmax = params['bound_lmax']
    quad_tol = params['sph_quad_tol']

    print("Testing potential energy matrix elements using Lebedev quadratures")

    sphlist = MAPPING.GEN_SPHLIST(lmax)

    val =  np.zeros( shape = ( len(sphlist)**2 ), dtype=complex)
    val_prev = np.zeros( shape = ( len(sphlist)**2 ), dtype=complex)

    spherical_schemes = []
    for elem in list(quadpy.u3.schemes.keys()):
        if 'lebedev' in elem:
            spherical_schemes.append(elem)

    xi = 0
    for i in range(np.size( rgrid, axis=0 )): 
        for n in range(np.size( rgrid, axis=1 )): 
            rin = rgrid[i,n]
            print("i = " + str(i) + ", n = " + str(n) + ", xi = " + str(xi) + ", r = " + str(rin) )
            if rin <= params['r_cutoff']:
                for scheme in spherical_schemes[3:]: #skip 003a,003b,003c rules
                    k=0
                    for l1,m1 in sphlist:
                        for l2,m2 in sphlist:
                        
                            val[k] = calc_potmatelem_quadpy( l1, m1, l2, m2, rin, scheme, esp_interpolant )
                            print(  '%4d %4d %4d %4d'%(l1,m1,l2,m2) + '%12.6f %12.6fi' % (val[k].real, val[k].imag) + \
                                    '%12.6f %12.6fi' % (val_prev[k].real, val_prev[k].imag) + \
                                    '%12.6f %12.6fi' % (np.abs(val[k].real-val_prev[k].real), np.abs(val[k].imag-val_prev[k].imag)) + \
                                    '%12.6f '%np.abs(val[k]-val_prev[k])  )
                            k += 1

                    diff = np.abs(val - val_prev)

                    if (np.any( diff > quad_tol )):
                        print( str(scheme) + " convergence not reached" ) 
                        for k in range(len(val_prev)):
                            val_prev[k] = val[k]
                    elif ( np.all( diff < quad_tol ) ):     
                        print( str(scheme) + " convergence reached !!!")
                        sph_quad_list.append([ i, n, xi, str(scheme)])
                        xi += 1
                        break

                    #if no convergence reached raise warning
                    if ( scheme == spherical_schemes[len(spherical_schemes)-1] and np.any( diff > quad_tol )):
                        print("WARNING: convergence at tolerance level = " + str(quad_tol) + " not reached for all considered quadrature schemes")

            else:
                scheme = spherical_schemes[lmax+1]
                sph_quad_list.append([ i, n, xi, str(scheme)])
                print("r>r_cutoff: "+str(scheme))
                xi += 1


    print("Converged quadrature levels: ")
    print(sph_quad_list)
    quadfilename = params['working_dir'] + params['file_quad_levels'] 
    fl = open(quadfilename,'w')
    print("saving quadrature levels to file: " + quadfilename )
    for item in sph_quad_list:
        fl.write( str('%4d '%item[0]) + str('%4d '%item[1]) + str('%4d '%item[2]) + str(item[3]) + "\n")

    return sph_quad_list

def plot_mat(mat):
    """ plot 2D array with color-coded magnitude"""
    fig, ax = plt.subplots()

    im, cbar = heatmap(mat, 0, 0, ax=ax, cmap="gnuplot", cbarlabel="Hij")

    fig.tight_layout()
    plt.show()

def heatmap( data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs ):
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
    im = ax.imshow(np.abs(data), **kwargs)

    # Create colorbar
    im.set_clim(0, np.max(np.abs(data)))
    cbar = ax.figure.colorbar(im, ax=ax,    **cbar_kw)
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


if __name__ == "__main__":      

    params = input.gen_input()
    BUILD_HMAT0(params)

