#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
from scipy import interpolate
from scipy.special import sph_harm

import GRID 

import os
import time
import json
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from multipoles import MultipoleExpansion


def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError

def sph2cart(r,theta,phi):
    x=r*np.sin(theta)*np.cos(phi)
    y=r*np.sin(theta)*np.sin(phi)
    z=r*np.cos(theta)
    return x,y,z


def slater(XYZ, xyz_mol, q_array):
        g = np.zeros_like(XYZ[0])
        for iatom in range(5):
            g += q_array[iatom] * np.exp(-1.0 * np.sqrt( (XYZ[0] - xyz_mol[0,iatom])**2 +\
                (XYZ[1] - xyz_mol[1,iatom])**2 +\
                (XYZ[2] - xyz_mol[2,iatom])**2 ) )
        return g



def delta_gaussian(XYZ, xyz_mol, Q_array):
    epsilon = 0.1
    g = np.zeros_like(XYZ[0])
    for iatom in range(5):
        g += Q_array[iatom] * np.exp(-1.0 * ( (XYZ[0] - xyz_mol[0,iatom])**2 +\
            (XYZ[1] - xyz_mol[1,iatom])**2 +\
                (XYZ[2] - xyz_mol[2,iatom])**2 ) / (2.0 * epsilon**2) )
    return g/(epsilon * np.sqrt(2.0 * np.pi))


def slater2D(XY, Z0, xyz_mol, q_array):
        g = np.zeros_like(XY[0])
        for iatom in range(5):
            g += q_array[iatom] * np.exp(-1.0 * np.sqrt( (XY[0] - xyz_mol[0,iatom])**2 +\
                (XY[1] - xyz_mol[1,iatom])**2 +\
                (Z0 - xyz_mol[2,iatom])**2 ) )
        return g



def delta_gaussian2D(XY,Z0, xyz_mol, Q_array):
    epsilon = 0.1
    g = np.zeros_like(XY[0])
    for iatom in range(5):
        g += Q_array[iatom] * np.exp(-1.0 * ( (XY[0] - xyz_mol[0,iatom])**2 +\
            (XY[1] - xyz_mol[1,iatom])**2 +\
                (Z0 - xyz_mol[2,iatom])**2 ) / (2.0 * epsilon**2) )
    return g/(epsilon * np.sqrt(2.0 * np.pi))


def chiralium_charge_distr(npoints,edge):
    """ Build charge distribution for the model chiralium molecule"""

    # 1. Positions of the nuclei given in spherical coordinates

    r_array     = np.array([4.0, 2.0, 2.0, 4.0, 0.0 ])
    theta_array = np.array([0.0, np.pi/2.0, np.pi/2.0, 2.19, 0.0])
    phi_array   = np.array([0.0, 0.0, np.pi/2.0, 3.93, 0.0])
    Q_array     = np.array([0.5, 1.5, 1.0, 0.25, 2.0])
    q_array     = np.array([-0.1, -1.0, -0.2, -0.45, -2.5])

    x,y,z = sph2cart(r_array,theta_array,phi_array)

    xyz_mol = np.vstack((x,y,z))

    x, y, z = [np.linspace(-edge/2., edge/2., npoints)]*3
    XYZ = np.meshgrid(x, y, z, indexing='ij')

    # Initialize the charge density rho, which is a 3D numpy array:
    rho = slater(XYZ, xyz_mol,q_array) + delta_gaussian(XYZ, xyz_mol,Q_array)

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    Z0 = -0.0
    XY = np.meshgrid(x,y, indexing='ij')
    rho2D = slater2D(XY,Z0, xyz_mol,q_array) + delta_gaussian2D(XY,Z0, xyz_mol,Q_array)
    img = ax.contourf(XY[0], XY[1], rho2D)
    
    fig.colorbar(img)
    plt.show()
    """
    #exit()


    return rho, XYZ

def calc_multipoles(params):

    qlm = []
    Lmax = params['multi_lmax']


    npoints = params['multi_ncube_points']
    edge    = params['multi_box_edge']

    rho, XYZ = chiralium_charge_distr(npoints,edge)

    charge_dist = {
        'discrete': False,
        'rho': rho,
        'xyz': XYZ
    }

    Phi = MultipoleExpansion(charge_dist, Lmax)
    qlm = Phi.multipole_moments

    """ Plot potential"""
    npoints2D = 200
    x, y = [np.linspace(-edge/2., edge/2., npoints2D)]*2
    XY = np.meshgrid(x,y, indexing='ij')

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    Z0 = 0.0
    rho2D = np.zeros((len(x),len(y)))
    XY = np.meshgrid(x,y, indexing='ij')
    for i in range(len(x)):
        for j in range(len(x)):
            rho2D[i,j] = Phi(x[i],y[j],Z0)
    print(rho2D.shape)
    print(rho2D)

    plot_cont_1 = ax.contourf( XY[0], XY[1], rho2D, 
                                20, 
                                cmap = 'jet', 
                                vmin = -20.0,
                                vmax = 20.0)

    fig.colorbar(plot_cont_1)
    plt.show()
    """
    


    with open(params['job_directory']+ "multipoles.dat", 'w') as qlmfile: 
        qlmfile.write( str(qlm) )
        #json.dump(qlm, qlmfile, indent=4, default=convert) #not suitable for tule keys

    return qlm


def read_potential(params):
    #read the partial waves representation of the electrostatic potential. A. Artemyev's potential.
    #vLM[xi, L, M]

    Nr = (params['bound_nlobs']-1) * params['bound_nbins']

    vLM = np.zeros((    Nr,
                        params['multi_lmax'] + 1, 
                        2 * params['multi_lmax'] + 1), 
                        dtype=complex)

    for L in range(params['multi_lmax'] + 1):
        for M in range(-L,L+1):
            if M == 0:
                suffix = ""
            elif M > 0:
                suffix = "_pos"
            elif M < 0:
                suffix = "_neg"

            vlmfile = open(params['main_dir'] + "potential/potential_L" + str(L) + "_M" + str(abs(M))+suffix + ".dat" , 'r' )
            v = []

            for _ in range(3):
                next(vlmfile) #skip header

            for line in vlmfile:
                words =  line.split()
                r     =  float(words[0])
                v_re  =  float(words[1])
                v_im  =  float(words[2])

                v.append([r, v_re + 1j * v_im])
            vlmfile.close()

            #v.append([400.0,0.0+1j*0.0])
            v = np.asarray(v)
            vLM[:,L,L+M] = v[:Nr,1] #assuming our grid matches the one for the potential!!!

    rgrid = v[:Nr,0]

    return vLM,rgrid

def INTERP_POT(params):
    #interpolate potential on the grid

    fl = open( params['working_dir'] + params['esp_file'] + ".dat", 'r' )
    esp = []

    for line in fl:
        words = line.split()
        x = float(words[0])
        y = float(words[1])
        z = float(words[2])
        v = float(words[3])
        esp.append([x,y,z,v])
        
    esp =  np.asarray(esp)  #NOTE: Psi4 returns ESP for a positive unit charge, so that the potential of a cation is positive. 
                            #We want ESP for negaitve unit charge, so we must change sign: attractive interaction.
    esp[:,3] = -1.0 * esp[:,3]

    start_time = time.time()
    esp_interp = interpolate.LinearNDInterpolator( ( esp[:,0], esp[:,1], esp[:,2] ) , esp[:,3])
    end_time = time.time()
    print("Interpolation of " + params['esp_file']  + " potential took " +  str("%10.3f"%(end_time-start_time)) + "s")
    
    if params['plot_esp'] == True:
        X = np.linspace(min(esp[:,0]), max(esp[:,0]),100)
        Y = np.linspace(min(esp[:,1]), max(esp[:,1]),100)
        X, Y = np.meshgrid(X, Y)  

        fig = plt.figure() 
        Z = esp_interp(X, 0, Y)
        plt.pcolormesh(X, Y, Z, shading = 'auto')
        plt.colorbar()
        plt.show()

    return esp_interp

def calc_interp_sph(interpolant,r,theta,phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return interpolant(x,y,z)

def BUILD_ESP_MAT(Gs,rgrid,esp_interpolant,r_cutoff):
    VG = []
    Gr = rgrid.flatten()

    for igs, gs in enumerate(Gs):
        if Gr[igs] <= r_cutoff:
            VG.append( np.array( [calc_interp_sph( esp_interpolant, Gr[igs], gs[:,0], gs[:,1] )] ) )
        else:
            VG.append( np.array( [-1.0 / Gr[igs] * np.ones(np.size(gs[:,0])) ] ) )
    return VG

def BUILD_ESP_MAT_EXACT(params, Gs, Gr):

    if os.path.isfile(params['working_dir'] + "esp/" + params['file_esp']):
        print (params['file_esp'] + " file exist")

        #os.remove(params['working_dir'] + "esp/" + params['file_esp'])

        if os.path.getsize(params['job_directory']  + "esp/" + params['file_esp']) == 0:

            print("But the file is empty.")
            os.remove(params['working_dir'] + "esp/" + params['file_esp'])
            os.remove(params['working_dir'] + "esp/grid.dat")

            grid_xyz = GRID.GEN_XYZ_GRID(Gs, Gr, params['job_directory'] + "esp/")
            grid_xyz = np.asarray(grid_xyz)
            V        = GRID.CALC_ESP_PSI4(params['job_directory'] + "esp/",params)
            V        = -1.0 * np.asarray(V)
            esp_grid = np.hstack((grid_xyz,V[:,None])) 
            fl       = open(params['job_directory']  + "esp/" + params['file_esp'],"w")
            np.savetxt(fl,esp_grid, fmt='%10.6f')

        else:
            print("The file is not empty.")
            flpot1 = open(params['working_dir'] + "esp/" + params['file_esp'], "r")
            V = []
            for line in flpot1:
                words   = line.split()
                potval  = float(words[3])
                V.append(potval)
            V = np.asarray(V)

    else:
        print (params['file_esp'] + " file does not exist")

        #os.remove(params['working_dir'] + "esp/grid.dat")

        grid_xyz = GRID.GEN_XYZ_GRID(Gs, Gr, params['working_dir'] + "esp/")
        grid_xyz = np.asarray(grid_xyz)
        V        = GRID.CALC_ESP_PSI4(params['working_dir'] + "esp/", params)
        V        = -1.0 * np.asarray(V)

        esp_grid = np.hstack((grid_xyz,V[:,None])) 
        fl       = open(params['working_dir'] + "esp/" + params['file_esp'], "w")
        np.savetxt(fl, esp_grid, fmt='%10.6f')

    r_array = Gr.flatten()

    VG = []
    counter = 0
    for k in range(len(r_array)-1):
        sph = np.zeros(Gs[k].shape[0], dtype=float)
        print("No. spherical quadrature points  = " + str(Gs[k].shape[0]) + " at grid point " + str(r_array[k]) )
        for s in range(Gs[k].shape[0]):
            sph[s] = V[counter]
            counter  += 1

        VG.append(sph)
    return VG


def BUILD_ESP_MAT_EXACT_ROT(params, Gs, Gr, mol_xyz, irun):


    r_array = Gr.flatten()

    VG = []
    counter = 0
    if params['molec_name'] == "chiralium": # test case for chiralium. We have analytic potential on the radial grid and we want to use Lebedev quadratures for matrix elements
        
        # 1. Read potential

        start_time = time.time()
        vLM,rgrid_anton = read_potential(params)
        end_time = time.time()

        lmax_multi = params['multi_lmax']

        for k in range(len(r_array)-1):
            sph = np.zeros(Gs[k].shape[0], dtype=complex)
            print("No. spherical quadrature points  = " + str(Gs[k].shape[0]) + " at grid point " + str('{:10.3f}'.format(r_array[k])))
            for s in range(Gs[k].shape[0]):

                #entire potential block 
                for L in range(0,lmax_multi+1):
                    for M in range(-L,L+1):
            
                        #Gs[k][s,0] = theta
                        sph[s] += vLM[k,L,L+M] * sph_harm( M, L, Gs[k][s,1]+np.pi, Gs[k][s,0] )
                counter  += 1

            VG.append(sph)


        return VG


    if os.path.isfile(params['job_directory']  + "esp/" +str(irun) + "/" + params['file_esp']):
        print (params['file_esp'] + " file exist")

        #os.remove(params['working_dir'] + "esp/" + params['file_esp'])

        if os.path.getsize(params['job_directory'] + "esp/" +str(irun) + "/"  + params['file_esp']) == 0:

            print("But the file is empty.")
            os.remove(params['job_directory'] + "esp/"+str(irun) + "/"  + params['file_esp'])
            os.remove(params['job_directory']  + "esp" +str(irun) + "/" +"grid.dat")

            grid_xyz = GRID.GEN_XYZ_GRID(Gs, Gr, params['job_directory']  + "esp/"+str(irun) + "/" )
            grid_xyz = np.asarray(grid_xyz)
            V        = GRID.CALC_ESP_PSI4(params['job_directory']  + "esp/"+str(irun) + "/" , params)
            V        = -1.0 * np.asarray(V)
            esp_grid = np.hstack((grid_xyz,V[:,None])) 
            fl       = open(params['job_directory']  + "esp/"+str(irun) + "/"  + params['file_esp'], "w")
            np.savetxt(fl,esp_grid, fmt='%10.6f')

        else:
            print("The file is not empty.")
            flpot1 = open(params['job_directory']  + "esp/" +str(irun) + "/" + params['file_esp'], "r")
            V = []
            for line in flpot1:
                words   = line.split()
                potval  = float(words[3])
                V.append(potval)
            V = np.asarray(V)

    else:
        print (params['file_esp'] + " file does not exist")

        #os.remove(params['working_dir'] + "esp/grid.dat")

        grid_xyz = GRID.GEN_XYZ_GRID(Gs, Gr, params['job_directory']  + "esp/"+str(irun) + "/" )
        grid_xyz = np.asarray(grid_xyz)
        V        = GRID.CALC_ESP_PSI4_ROT(params['job_directory']  + "esp/"+str(irun) + "/" , params, mol_xyz)
        V        = -1.0 * np.asarray(V)

        esp_grid = np.hstack((grid_xyz,V[:,None])) 
        fl       = open(params['job_directory']  + "esp/"+str(irun) + "/"  + params['file_esp'] + "_"+str(irun), "w")
        np.savetxt(fl, esp_grid, fmt='%10.6f')

    r_array = Gr.flatten()

    VG = []
    counter = 0
    if params['molec_name'] == "h": # test case of shifted hydrogen
        r0 = 1.0
        for k in range(len(r_array)-1):
            sph = np.zeros(Gs[k].shape[0], dtype=float)
            print("No. spherical quadrature points  = " + str(Gs[k].shape[0]) + " at grid point " + str('{:10.3f}'.format(r_array[k])))
            for s in range(Gs[k].shape[0]):
                sph[s] = -1.0 / np.sqrt(r_array[k]**2 + r0**2 - 2.0 * r_array[k] * r0 * np.cos(Gs[k][s,0]))
                #sph[s] = -1.0 / (r_array[k])
                counter  += 1

            VG.append(sph)

    else:
        for k in range(len(r_array)-1):
            sph = np.zeros(Gs[k].shape[0], dtype=float)
            print("No. spherical quadrature points  = " + str(Gs[k].shape[0]) + " at grid point " + str('{:10.3f}'.format(r_array[k])) )
            for s in range(Gs[k].shape[0]):
                sph[s] = V[counter]
                counter  += 1

            VG.append(sph)

    return VG