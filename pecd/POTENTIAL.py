#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
from scipy import interpolate

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

    fl = open( params['working_dir'] +  params['esp_file']  + ".dat", 'r' )
    esp = []

    for line in fl:
        words = line.split()
        x = float(words[0])
        y = float(words[1])
        z = float(words[2])
        v = -1.0 * float(words[3])
        esp.append(v)
        
    r_array = Gr.flatten()

    VG = []
    counter = 0
    for k in range(len(r_array)):
        print(Gs[k].shape[0])
        sph = np.zeros(Gs[k].shape[0],dtype=float)

        for s in range(Gs[k].shape[0]):
            sph[s] = esp[counter]
            counter  += 1

        VG.append(sph)
    return VG