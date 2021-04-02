#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np

def GENMAP(nlobs,nbins,lmax,maptype,working_dir):

    if maptype == 'DVR':
        maparray, Nbas = MAP_DVR(nlobs,nbins,lmax)
    elif maptype == 'SPECT':
        maparray, Nbas = MAP_SPECT(nlobs,nbins,lmax)

    fl = open(working_dir + 'map.dat','w')
    for elem in maparray:   
        fl.write("%5d"%elem[0]+"  %5d"%elem[1]+ "  %5d"%elem[2]+ "  %5d"%elem[3]+  " %5d"%elem[4]+" %5d"%elem[5]+"\n")
    fl.close()


    return maparray, Nbas


def MAP_DVR(nlobs,nbins,lmax):

    imap = 0
    xi = 0
    maparray = []

    for i in range(0,nbins):
        for n in range (0,nlobs):
            if n == nlobs-1:
                continue        
            elif n == 0 and i == nbins-1:
                continue
            else:
                xi += 1
                for l in range(0,lmax+1):
                    for m in range(-l,l+1):

                        imap += 1
                        print(i,n,xi,l,m,imap)
                        maparray.append([i,n,xi,l,m,imap])
            

    return maparray, imap


def MAP_SPECT(nlobs,nbins,lmax):

    imap = 0
    maparray = []

    for l in range(0,lmax+1):
        for m in range(-l,l+1):
            xi = 0
            for i in range(0,nbins):
                for n in range (0,nlobs):
                    if n == nlobs-1:
                        continue        
                    elif n == 0 and i == nbins-1:
                        continue
                    else:
                        xi += 1
                        imap += 1
                        print(l,m,i,n,xi,imap)
                        maparray.append([l,m,i,n,xi,imap])
            

    return maparray, imap


def GEN_SPHLIST(lmax):
    #create list of basis set indices for adaptive quadratures
    sphlist = []
    if lmax == 0:
        raise ValueError("lmax = 0 makes no sense in the generation of adaptive quadratures")

    for l in range(lmax-1,lmax+1):
        for m in range(l-1,l+1):
            sphlist.append([l,m])
    return sphlist

""" TESTING 
params = {}
params['map_type'] = 'SPECT'
params['working_dir'] = "/Users/zakemil/Nextcloud/projects/PECD/tests/molecules/h2o/"
GENMAP(5,3,1,params)
"""