#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
import sys
print(sys.version)
import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

class angbas():
    """comment"""

    def __init__(self):
        pass
    
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

    def spharm(self,l,m,theta,phi):
        return sph_harm(m, l, phi, theta)

    def spharm_rot(self,l,m,phi,theta,alpha,beta,zgamma):
        #return rotated spherical harmonic in active transformation from MF to LAB frame. 
        Yrot=0.0
        #print("value of the wigner D matrix: ", Rotation.D(l, m, 0 , alpha, beta ,zgamma).doit().evalf())
    
        for mu in range(-l,l+1):
            #print(l,mu,m)
            Yrot+=complex(Rotation.D(l, m, mu , alpha, beta, zgamma).doit().evalf())*spharm(l, mu, phi, theta)
        return Yrot

    def solidharm(self,l,m,r,theta,phi):
    
        if m==0:
            SH=np.sqrt(4.0*np.pi/(2.0*float(l)+1))*(r**l)*np.real(scipy.special.sph_harm(0, l, phi, theta))
        elif m>0:
            SH=((-1.0)**m)*np.sqrt(8.0*np.pi/(2.0*float(l)+1))*(r**l)*np.real(scipy.special.sph_harm(m, l, phi, theta))
        elif m<0:
            SH=((-1.0)**m)*np.sqrt(8.0*np.pi/(2.0*float(l)+1))*(r**l)*np.imag(scipy.special.sph_harm(-m, l, phi, theta))
        return SH

    def solidharm_vec(self,l,m,r,theta,phi):
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


    def show_Y_lm(self,l, m):
        """plot the angular basis"""
        theta_1d = np.linspace(0,   np.pi,  91) # 2 GRAD Schritte
        phi_1d   = np.linspace(0, 2*np.pi, 181) # 2 GRAD Schritte

        theta_2d, phi_2d = np.meshgrid(theta_1d, phi_1d)
        xyz_2d = np.array([np.sin(theta_2d) * np.sin(phi_2d),
                        np.sin(theta_2d) * np.cos(phi_2d),
                        np.cos(theta_2d)]) 

        colormap = cm.ScalarMappable( cmap=plt.get_cmap("cool"))
        colormap.set_clim(-.45, .45)
        limit = .5

        print("Y_%i_%i" % (l,m)) # zeigen, dass was passiert
        plt.figure()
        ax = plt.gca(projection = "3d")
        
        plt.title("$Y^{%i}_{%i}$" % (m,l))
        Y_lm = sph_harm(m,l, phi_2d, theta_2d)
        r = np.abs(Y_lm.real)*xyz_2d
        ax.plot_surface(r[0], r[1], r[2], facecolors=colormap.to_rgba(Y_lm.real), rstride=1, cstride=1)
        ax.set_xlim(-limit,limit)
        ax.set_ylim(-limit,limit)
        ax.set_zlim(-limit,limit)
        #ax.set_aspect("equal")
        #ax.set_axis_off()
        
                
        plt.show()


if __name__ == "__main__":      
    print("hello")

    Ylm = angbas()
    #Ylm.show_Y_lm(l=5,m=0)
    #Ylm.show_Y_lm(l=5,m=4)        
    Ylm.show_Y_lm(l=2,m=2)
