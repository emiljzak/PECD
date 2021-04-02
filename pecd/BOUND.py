#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#

import input
import MAPPING
import POTENTIAL

import numpy as np
from scipy import sparse

import sys
import time


def BUILD_HMAT0(params):

    maparray, Nbas = MAPPING.GENMAP(params['bound_nlobs'],params['bound_nbins'],params['bound_lmax'],\
        params['map_type'],params['working_dir'])

    if params['hmat_format'] == 'csr':
        hmat = sparse.csr_matrix((Nbas, Nbas), dtype=np.float64)
    elif params['hmat_format'] == 'regular':
        hmat = np.zeros((Nbas, Nbas), dtype=np.float64)


    """ calculate hmat """
    BUILD_POTMAT0(params)

    """ diagonalize hmat """
    start_time = time.time()
    enr0, coeffs0 = np.linalg.eigh(hmat, UPLO = 'U')
    end_time = time.time()
    print("Time for diagonalization of field-free Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")


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
                            "\t ".join('{:10.5e}'.format(coeffs0[ielem,v]) for v in range(0,params['num_ini_vec'])) + "\n")

    if params['save_enr0'] == True:
        with open(params['working_dir'] + params['file_enr0'], "w") as energyfile:   
            np.savetxt( energyfile, enr0 , fmt='%10.5f' )
  
    print(hmat)

def BUILD_KEOMAT0(params):
    print("under construction")


def BUILD_POTMAT0(params):

    print("Interpolating electrostatic potential")
    esp_interpolant = POTENTIAL.INTERP_POT(params)

    if  params['gen_adaptive_quads'] == True:
        sph_quad_list = gen_adaptive_quads(params,esp_interpolant)






def gen_adaptive_quads(params,esp_interpolant):

    sph_quad_list = [] #list of quadrature levels needed to achieve convergence quad_tol of the potential energy matrix elements (global)

    lmax = params['lmax']
    quad_tol = params['sph_quad_tol']

    print("Testing potential energy matrix elements using Lebedev quadratures")
    #create list of basis set indices
    anglist = []
    for l in range(lmax-1,lmax+1):
        for m in range(l-1,l+1):
            anglist.append([l,m])

    val =  np.zeros(shape=(len(anglist)**2),dtype=complex)
    val_prev = np.zeros(shape=(len(anglist)**2),dtype=complex)

    #pull out Lebedev schemes into a list
    spherical_schemes = []
    for elem in list(quadpy.u3.schemes.keys()):
        if 'lebedev' in elem:
            spherical_schemes.append(elem)
    #print("Available schemes: " + str(spherical_schemes))

    for i in range(np.size(self.rgrid,axis=0)): #iterate over radial indices in the basis: only l=0, m=0
        for n in range(np.size(self.rgrid,axis=1)): 
            rin = self.rgrid[i,n]
            print(i,n,rin)
            if rin <= self.params['r_cutoff']:
                #iterate over the schemes
                for scheme in spherical_schemes[3:]: #skip 003a,003b,003c rules

                    k=0
                    for l1,m1 in anglist:
                        for l2,m2 in anglist:
                        
                            val[k] = self.calc_potmatelem_interp_vec(l1,m1,l2,m2,rin,scheme,esp_interpolant )
                            print(" %4d %4d %4d %4d"%(l1,m1,l2,m2) + '%12.6f %12.6fi' % (val[k].real, val[k].imag)+ '%12.6f %12.6fi' % (val_prev[k].real, val_prev[k].imag) + \
                                '%12.6f %12.6fi' % (np.abs(val[k].real-val_prev[k].real), np.abs(val[k].imag-val_prev[k].imag)) + '%12.6f '%np.abs(val[k]-val_prev[k])  )
                            k+=1

                    #check for convergence
                    diff = np.abs(val - val_prev)

                    if (np.any(diff>quad_tol)):
                        print(str(scheme)+" convergence not reached") 
                        for k in range(len(val_prev)):
                            val_prev[k] = val[k]

                    elif (np.all(diff<quad_tol)):     
                        print(str(scheme)+" convergence reached!!!")
                        val_conv = val_prev
                        levels.append([i,n,str(scheme)])
                        break

                    #if no convergence reached raise warning
                    if (scheme == spherical_schemes[len(spherical_schemes)-1] and np.any(diff>quad_tol)):
                        print("WARNING: convergence at tolerance level = " + str(quad_tol) + " not reached for all considered quadrature schemes")
            else:
                scheme == spherical_schemes[0]
                levels.append([i,n,str(scheme)])

    print("Converged quadrature levels:")
    print(levels)
    quadfilename = self.params['working_dir']+self.params['quad_levels_file']
    fname=open(quadfilename,'w')
    print("saving quadrature levels to file: " + quadfilename )
    for item in levels:
        fname.write(str('%4d '%item[0]) +str('%4d '%item[1])+str(item[2]) +"\n")



if __name__ == "__main__":      

    params = input.gen_input()
    BUILD_HMAT0(params)

