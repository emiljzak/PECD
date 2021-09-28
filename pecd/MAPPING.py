#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
from numba import jit, prange
jitcache = False

def GENMAP_FEMLIST(femlist,lmax,maptype,working_dir):
    #generate mapping for general FEMLIST
    if maptype == 'DVR':
        maparray, Nbas = MAP_DVR_FEMLIST_NAT(femlist,lmax)
    elif maptype == 'SPECT':
        maparray, Nbas = MAP_SPECT_FEMLIST(femlist,lmax)

    #fl = open(working_dir + 'map.dat','w')
    #for elem in maparray:   
    #    fl.write("%5d"%elem[0]+"  %5d"%elem[1]+ "  %5d"%elem[2]+ "  %5d"%elem[3]+  " %5d"%elem[4]+" %5d"%elem[5]+"\n")
    #fl.close()


    return maparray, Nbas



def MAP_DVR_FEMLIST(femlist,lmax):
    imap = 0
    xi = 0
    maparray = []
    ibincount = -1

    nbins = 0
    for elem in femlist:
        nbins += elem[0]
    print("total number of bins = " + str(nbins))

    for elem in femlist:
        for i in range(elem[0]):
            ibincount +=1
            for n in range(elem[1]):
                if n == elem[1]-1:
                    continue   
                elif n == 0 and ibincount == nbins-1:
                    continue     
                else:
                    xi += 1
                    for l in range(0,lmax+1):
                        for m in range(-l,l+1):

                            imap += 1
                            #print(ibincount,n,xi,l,m,imap)
                            maparray.append([ibincount,n,xi,l,m,imap])


    return maparray, imap


def MAP_DVR_FEMLIST_NAT(femlist,lmax):
    #natural order of grid points and basis functions, including bridges
    imap = 0
    xi = 0
    maparray = []
    ibincount = -1

    nbins = 0
    for elem in femlist:
        nbins += elem[0]
    print("total number of bins = " + str(nbins))

    for elem in femlist:
        for i in range(elem[0]):
            ibincount +=1
            for n in range(1,elem[1]):
                if ibincount == nbins-1 and n == elem[1]-1:
                    continue     
                else:
                    xi += 1
                    for l in range(0,lmax+1):
                        for m in range(-l,l+1):

                            imap += 1
                            #print(ibincount,n,xi,l,m,imap)
                            maparray.append([ibincount,n,xi,l,m,imap])


    return maparray, imap



def MAP_SPECT_FEMLIST(femlist,lmax):
    ##### NEEDS VERIFICATION #########
    imap = 0
    xi = 0
    maparray = []
    ibincount = -1

    nbins = 0
    for elem in femlist:
        nbins += elem[0]
    print("total number of bins = " + str(nbins))
    
    for l in range(0,lmax+1):
        for m in range(-l,l+1):
            xi = 0
            ibinscound = -1
            for elem in femlist:
                for i in range(elem[0]):
                    ibincount +=1
                    for n in range(elem[1]):
                        if n == elem[1]-1:
                            continue   
                        elif n == 0 and ibincount == nbins-1:
                            continue     
                        else:
                            xi += 1
                            imap += 1
                            #print(l,m,ibincount,n,xi,imap)
                            maparray.append([l,m,ibincount,n,xi,imap])

    return maparray, imap




def GENMAP(nlobs,nbins,lmax,maptype,working_dir):
    #only for uniform FEMLIST of bins
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
                        #print(i,n,xi,l,m,imap)
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
                        #print(l,m,i,n,xi,imap)
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


#@jit( nopython=True, parallel=False, cache = jitcache, fastmath=False) 
def GEN_VLIST(maparray, Nbas, map_type):
    #create list of indices for matrix elements of the potential
    vlist = []

    #needs speed-up: convert maparray to numpy array and vectorize, add numpy

    if map_type == 'DVR':
        for p1 in range(Nbas):
            for p2 in range(p1, Nbas):
                if maparray[p1][2] == maparray[p2][2]: #and maparray[p1][2] < xi_cutoff
                    vlist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], \
                        maparray[p2][3], maparray[p2][4], p1, p2  ])

    elif map_type == 'SPECT':
        for p1 in range(Nbas):
            for p2 in range(p1, Nbas):
                if maparray[p1][2] == maparray[p2][2]:
                    vlist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], maparray[p2][3], maparray[p2][4] ])
    return vlist


def GEN_KLIST(maparray, Nbas, map_type):
    #create list of indices for matrix elements of the KEO
    klist = []

    if map_type == 'DVR':
        for p1 in range(Nbas):
            for p2 in range(p1, Nbas):
                if maparray[p1][3] == maparray[p2][3] and maparray[p1][4] == maparray[p2][4]:
                    if maparray[p1][0] == maparray[p2][0] or maparray[p1][0] == maparray[p2][0] - 1 or maparray[p1][0] == maparray[p2][0] + 1: 
                        klist.append([ maparray[p1][0], maparray[p1][1], maparray[p2][0], \
                        maparray[p2][1], maparray[p2][3], p1, p2 ])


    return klist


""" TESTING 
params = {}
params['map_type'] = 'SPECT'
params['working_dir'] = "/Users/zakemil/Nextcloud/projects/PECD/tests/molecules/h2o/"
GENMAP(5,3,1,params)
"""