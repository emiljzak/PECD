#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

class angbas():
    """comment"""

    def __init__(self):
        pass
    
    def plot_bas(self):
        """plot the angular basis"""
        phi = np.linspace(0, np.pi, 100)
        theta = np.linspace(0, 2*np.pi, 100)
        phi, theta = np.meshgrid(phi, theta)

        # The Cartesian coordinates of the unit sphere
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        m, l = 2, 3

        # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
        fcolors = sph_harm(m, l, theta, phi).real
        fmax, fmin = fcolors.max(), fcolors.min()
        fcolors = (fcolors - fmin)/(fmax - fmin)

        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=plt.figaspect(1.))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.seismic(fcolors))
        # Turn off the axis planes
        ax.set_axis_off()
        plt.show()

    def orth_test(self):
        """test orthogonality relations for the basis set"""

    def sph_cart(self,r,theta,phi):
        x=r*np.sin(theta)*np.cos(phi)
        y=r*np.sin(theta)*np.sin(phi)
        z=r*np.cos(theta)
        return x,y,z

    def cart_sph(self,x,y,z):
        r=np.sqrt(x**2+y**2+z**2)
        theta=np.arctan(np.sqrt(x**2+y**2)/z)
        phi=np.arctan(y/x)
        return r,theta,phi

    def spharm(l,m,phi,theta):
        return scipy.special.sph_harm(m, l, phi, theta)

    def spharm_rot(l,m,phi,theta,alpha,beta,zgamma):
        #return rotated spherical harmonic in active transformation from MF to LAB frame. 
        Yrot=0.0
        #print("value of the wigner D matrix: ", Rotation.D(l, m, 0 , alpha, beta ,zgamma).doit().evalf())
    
        for mu in range(-l,l+1):
            #print(l,mu,m)
            Yrot+=complex(Rotation.D(l, m, mu , alpha, beta, zgamma).doit().evalf())*spharm(l, mu, phi, theta)
        return Yrot

    def solidharm(l,m,r,theta,phi):
    
        if m==0:
            SH=np.sqrt(4.0*np.pi/(2.0*float(l)+1))*(r**l)*np.real(scipy.special.sph_harm(0, l, phi, theta))
        elif m>0:
            SH=((-1.0)**m)*np.sqrt(8.0*np.pi/(2.0*float(l)+1))*(r**l)*np.real(scipy.special.sph_harm(m, l, phi, theta))
        elif m<0:
            SH=((-1.0)**m)*np.sqrt(8.0*np.pi/(2.0*float(l)+1))*(r**l)*np.imag(scipy.special.sph_harm(-m, l, phi, theta))
        return SH

    def solidharm_vec(l,m,r,theta,phi):
        #print(theta.size)
        SH=np.array([0.0]*theta.size)
        if m==0:
            for i in range(0,theta.size):
                SH[i]=np.sqrt(4.0*np.pi/(2.0*float(l)+1))*(r**l)*scipy.special.sph_harm(l, 0, theta[i], phi[i])
        elif m>0:
            for i in range(0,theta.size):
                SH[i]=((-1.0)**m)*np.sqrt(8.0*np.pi/(2.0*float(l)+1))*(r**l)*np.real(scipy.special.sph_harm(l, m, theta[i], phi[i]))
        elif m<0:
            for i in range(0,theta.size):
                SH[i]=((-1.0)**m)*np.sqrt(8.0*np.pi/(2.0*float(l)+1))*(r**l)*np.imag(scipy.special.sph_harm(l, -m, theta[i], phi[i]))
        return SH



#Ylm = angbas()
#plot_bas(Ylm)
print("hello")