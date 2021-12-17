#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
from logging import raiseExceptions
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
    #create a list of indices for matrix elements of the potential
    vlist = []

    #needs speed-up: convert maparray to numpy array and vectorize, add numpy

    if map_type == 'DVR':
        for p1 in range(Nbas):
            for p2 in range(p1, Nbas):
                if maparray[p1][2] == maparray[p2][2]: 
                    vlist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], \
                        maparray[p2][3], maparray[p2][4], p1, p2  ])

    elif map_type == 'SPECT':
        for p1 in range(Nbas):
            for p2 in range(p1, Nbas):
                if maparray[p1][2] == maparray[p2][2]:
                    vlist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], maparray[p2][3], maparray[p2][4] ])
    return vlist


def GEN_KLIST(maparray, Nbas, map_type):
    #create a list of indices for matrix elements of the KEO
    klist = []

    if map_type == 'DVR':
        for p1 in range(Nbas):
            for p2 in range(p1, Nbas):
                if maparray[p1][3] == maparray[p2][3] and maparray[p1][4] == maparray[p2][4]:
                    if maparray[p1][0] == maparray[p2][0] or maparray[p1][0] == maparray[p2][0] - 1 or maparray[p1][0] == maparray[p2][0] + 1: 
                        klist.append([ maparray[p1][0], maparray[p1][1], maparray[p2][0], \
                        maparray[p2][1], maparray[p2][3], p1, p2 ])


    return klist

def calc_p2(l,m,xi,lmax):
    return (xi-1)*(lmax+1)**2 + l*(l+1) + m


def GEN_DIPLIST_opt1(maparray, Nbas, lmax, map_type, sigma ):
    """
    create a list of indices for matrix elements of the dipole interaction matrix.     
    """
    
    diplist = []
    if map_type == 'DVR':
        for p1 in range(Nbas):

            xi = maparray[p1][2] 
            l1 = maparray[p1][3]
            m1 = maparray[p1][4]

            if l1+1 <= lmax:
                p2 = calc_p2(l1+1,m1-sigma,xi,lmax)
                diplist.append([ xi, l1, m1, l1+1, m1-sigma, p1, p2 ])
            if l1-1 >= 0:
                p2 = calc_p2(l1-1,m1-sigma,xi,lmax)
                diplist.append([ xi, l1, m1, l1-1, m1-sigma, p1, p2 ])
    else:
        ValueError("Incorrect map type")



    return diplist


def GEN_DIPLIST(maparray, Nbas, map_type, sigma ):
    """
    Old O(n**2) implementation.
    create a list of indices for matrix elements of the dipole interaction matrix. 
    Generates full square matrix for sigma = -1 or 0. 
    sigma = +1 can be generated using symmetries.
    
    """
    
    diplist = []
    if map_type == 'DVR':

        #set up indices for block-listing

        if sigma == 0:

            for p1 in range(Nbas):
                for p2 in range(Nbas):
                    if maparray[p1][2] == maparray[p2][2]: 

                        if maparray[p1][3] == maparray[p2][3] - 1 or maparray[p1][3] == maparray[p2][3] + 1:
                            if maparray[p1][4] == maparray[p2][4]: 
                                diplist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], \
                                maparray[p2][3], maparray[p2][4], p1, p2 ])

        elif sigma == +1:

            #for xi in range(Nr-1):
            #    for i1 in range(block_size):
            for p1 in range(Nbas):          
                for p2 in range(Nbas):
                    if maparray[p1][2] == maparray[p2][2]: 

                        if maparray[p1][3] == maparray[p2][3] - 1 or maparray[p1][3] == maparray[p2][3] + 1:
                            if maparray[p2][4] == maparray[p1][4] + 1: 
                                diplist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], \
                                maparray[p2][3], maparray[p2][4], p1, p2 ])

        elif sigma == -1:
            for p1 in range(Nbas):
                for p2 in range(Nbas):
                    if maparray[p1][2] == maparray[p2][2]: 

                        if maparray[p1][3] == maparray[p2][3] - 1 or maparray[p1][3] == maparray[p2][3] + 1:
                            if maparray[p2][4] == maparray[p1][4] - 1: 
                                diplist.append([ maparray[p1][2], maparray[p1][3], maparray[p1][4], \
                                maparray[p2][3], maparray[p2][4], p1, p2 ])
    else:
        ValueError("Incorrect map type")



    return diplist


