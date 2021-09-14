#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
from scipy import interpolate

import GRID 

import os
import time
import json
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

    def slater(XYZ, xyz_mol):
        g = np.zeros_like(XYZ[0])
        print(g.shape)
        for iatom in range(5):
            g += np.exp(-1.0 * np.sqrt( (XYZ[0] - xyz_mol[0,iatom])**2 +\
                (XYZ[1] - xyz_mol[1,iatom])**2 +\
                    (XYZ[2] - xyz_mol[2,iatom])**2 ) )
        return g

    # Initialize the charge density rho, which is a 3D numpy array:
    rho = slater(XYZ, xyz_mol)

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
    x, y, z = 2,2,2
    value = Phi(x, y, z)
    print(value)


    with open(params['job_directory']+ "multipoles.dat", 'w') as qlmfile: 
        json.dump(qlm, qlmfile, indent=4, default=convert)


    return qlm

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
    if params['molec_name'] == "h":
        for k in range(len(r_array)-1):
            sph = np.zeros(Gs[k].shape[0], dtype=float)
            print("No. spherical quadrature points  = " + str(Gs[k].shape[0]) + " at grid point " + str('{:10.3f}'.format(r_array[k])))
            for s in range(Gs[k].shape[0]):
                sph[s] = -1.0 / (r_array[k])
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