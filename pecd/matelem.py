#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
#import sys
#print(sys.version)#
import numpy as np
from basis import angbas
import quadpy
from scipy.special import sph_harm
print(quadpy.__version__)


class integration():
    """Class containing all integration methods"""

    def __init__(self):
        pass

    def test_angular_int(self,lmin,lmax,hdeg):
        """ test the accuracy of angular integration (lebedev quadrature) over the spherical harmonics basis"""
        anglist = []
        for l in range(lmin,lmax+1):
            for m in range(0,l+1):
                anglist.append([l,m])

        scheme = quadpy.u3.get_good_scheme(hdeg)
        #print(scheme.points, scheme.weights, scheme.degree, scheme.source, scheme.test_tolerance)
        print(" %4s %4s %4s %4s"%("l1","m1","l2","m2") + " %15s"%"val")

        #print(np.shape(scheme.points))
        #print(scheme.points[1]*180.0/np.pi)
        for l1,m1 in anglist:
            for l2,m2 in anglist:
                val = scheme.integrate_spherical(lambda theta_phi: np.conjugate(sph_harm(m1,l1,theta_phi[1],theta_phi[0])) * sph_harm(m2,l2,theta_phi[1],theta_phi[0]))
                print(" %4d %4d %4d %4d"%(l1,m1,l2,m2) + " %20.12f"%val)
            



class potential():
    """Class containing methods for the representation of the PES"""

    def __init__(self):
        pass

    def pot_sph(self,r):
        #spherical test potential
        #partial charges in OCS
        deltaC=0.177e0
        deltaO=-0.226e0
        deltaS=0.049e0
        
        rOC=115.78 #pm
        rCS=156.01 #pm

        pot=-1.0/r
        return pot


    def pot_ocs_pc(self,r,theta):
        #point charges from rigid OCS
        #partial charges in OCS
        #print(np.shape(theta))
        deltaC=0.177+0.3333
        deltaO=-0.226+0.3333
        deltaS=0.049+0.3333
        #OCS is aligned along z-axis. The potential is therefore phi-symmetric
        rOC=115.78 #pm
        rCS=156.01 #pm
        rOC/=100.0 #to angstrom
        rCS/=100.0

        rO=np.sqrt(rOC**2+r**2+2.0*rOC*r*np.cos(theta))
        rS=np.sqrt(rCS**2+r**2-2.0*rCS*r*np.cos(theta))
        
        #x=r*np.cos(theta)
        #y=r*np.sin(theta)
        #Veff=-(deltaC/r+deltaO/np.sqrt((x+rOC)**2+y**2+x**2)+deltaS/np.sqrt((x-rCS)**2+y**2+x**2)) #cartesian projection method
        pot = - (deltaC/r + deltaO/rO + deltaS/rS)
        return pot

    def pot_test(self,r,theta,phi):
        """test potential for quadrature integration"""
        return np.cos(theta)**2


    def plot_pot_2D(self):
        """plot the potential"""


class potmat(potential,integration):
    """Class containing methods for the calculation of the potential matrix elements"""

    def __init__(self,Hang):
        self.Hang  = Hang #degree H of angular (ang) quadrature (Lebedev)
    
    def generate(self):
        """calculate full potential matrix"""
        print("hello")


    def spharmcart(self,l,m,x):
        tol = 1e-8
        #print(np.shape(x[0]))
        if all(np.abs(x[0])>tol) and all(np.abs(x[2])>tol): 
            r=np.sqrt(x[0]**2+x[1]**2+x[2]**2)
            theta=np.arctan(np.sqrt(x[0]**2+x[1]**2)/x[2])
            phi=np.arctan(x[1]/x[0])
        else:
            r = 0
            theta = 0.0
            phi = 0.0
        #print(theta,phi)
        return sph_harm(m, l, phi, theta)


    def jacobian(self,x):
        theta=np.arctan(np.sqrt(x[0]**2+x[1]**2)/x[2])
        return np.sin(theta) 


    def pot_integrand(self,theta_phi):
        theta, phi = theta_phi
        return np.conjugate(sph_harm(m1,l1,phi,theta)) * sph_harm(m2,l2,phi,theta)



    def calc_potmatelem(self,l1,m1,l2,m2,rin):
        """calculate single element of potential matrix"""
        scheme = quadpy.u3.get_good_scheme(47)
        val = scheme.integrate_spherical(self.pot_integrand) 
        return val
    





if __name__ == "__main__":      


   # potmatrix.calc_potmatelem(l1=0,m1=0,l2=1,m2=1,0.0)

    potmatrix = integration()

    lmin = 0
    lmax = 2
    hdeg = 47
    potmatrix.test_angular_int(lmin,lmax,hdeg)

