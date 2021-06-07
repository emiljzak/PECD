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

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D


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

        if os.path.getsize(params['working_dir'] + "esp/" + params['file_esp']) == 0:

            print("But the file is empty.")
            os.remove(params['working_dir'] + "esp/" + params['file_esp'])
            os.remove(params['working_dir'] + "esp/grid.dat")

            grid_xyz = GRID.GEN_XYZ_GRID(Gs, Gr, params['working_dir'] + "esp/")
            grid_xyz = np.asarray(grid_xyz)
            V        = GRID.CALC_ESP_PSI4(params['working_dir'] + "esp/",params)
            V        = -1.0 * np.asarray(V)
            esp_grid = np.hstack((grid_xyz,V[:,None])) 
            fl       = open(params['working_dir'] + "esp/" + params['file_esp'],"w")
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


    if os.path.isfile(params['working_dir'] + "esp/" + params['file_esp']):
        print (params['file_esp'] + " file exist")

        #os.remove(params['working_dir'] + "esp/" + params['file_esp'])

        if os.path.getsize(params['working_dir'] + "esp/" + params['file_esp']) == 0:

            print("But the file is empty.")
            os.remove(params['working_dir'] + "esp/" + params['file_esp'])
            os.remove(params['working_dir'] + "esp/grid.dat")

            grid_xyz = GRID.GEN_XYZ_GRID(Gs, Gr, params['working_dir'] + "esp/")
            grid_xyz = np.asarray(grid_xyz)
            V        = GRID.CALC_ESP_PSI4(params['working_dir'] + "esp/",params)
            V        = -1.0 * np.asarray(V)
            esp_grid = np.hstack((grid_xyz,V[:,None])) 
            fl       = open(params['working_dir'] + "esp/" + params['file_esp'],"w")
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
        V        = GRID.CALC_ESP_PSI4_ROT(params['working_dir'] + "esp/", params, mol_xyz)
        V        = -1.0 * np.asarray(V)

        esp_grid = np.hstack((grid_xyz,V[:,None])) 
        fl       = open(params['working_dir'] + "esp/" + params['file_esp'] + "_"+str(irun), "w")
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