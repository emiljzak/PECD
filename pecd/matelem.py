#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
from basis import angbas
import quadpy
from scipy.special import sph_harm
#print(quadpy.__version__)


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


class potmat(potential):
    """Class containing methods for the calculation of the potential matrix elements"""

    def __init__(self,scheme):
        self.scheme  =  scheme #name of the quadrature scheme used (Lebedev)
    
    def calc_mat(self,lmin,lmax,rin):
        """calculate full potential matrix"""
        print("hello")
        #create list of basis set indices
        anglist = []
        for l in range(lmin,lmax+1):
            for m in range(0,l+1):
                anglist.append([l,m])

        #calc_potmatelem(self,l1,m1,l2,m2,rin,scheme)


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

    def calc_potmatelem(self,l1,m1,l2,m2,rin,scheme):
        """calculate single element of potential matrix"""
        #print(self.Hdeg)
        #print(self.scheme)
        myscheme = quadpy.u3.schemes[scheme]()

        #print(myscheme)
        val = myscheme.integrate_spherical(lambda theta_phi: np.conjugate(sph_harm(m1, l1, theta_phi[1], theta_phi[0])) * sph_harm(m2, l2, theta_phi[1], theta_phi[0]) )#* self.pot_ocs_pc(rin,theta_phi[0]) )
        return val
    


    def test_angular_convergence(self,lmin,lmax,quad_tol):
        
        #create list of basis set indices
        anglist = []
        for l in range(lmin,lmax+1):
            for m in range(0,l+1):
                anglist.append([l,m])


        #convergence test over matrix elements:
        val =  np.zeros(shape=(len(anglist)**2))
        val_prev = np.zeros(shape=(len(anglist)**2))


        #pull out Lebedev schemes into a list
        spherical_schemes = []
        for elem in list(quadpy.u3.schemes.keys()):
            if 'lebedev' in elem:
                spherical_schemes.append(elem)
        print("Available schemes: " + str(spherical_schemes))

        #iterate over the schemes
        for scheme in spherical_schemes[3:]: #skip 003a,003b,003c rules

            i=0
            for l1,m1 in anglist:
                for l2,m2 in anglist:
                
                    val[i] = self.calc_potmatelem(l1,m1,l2,m2,1.0,scheme)
                    print(" %4d %4d %4d %4d"%(l1,m1,l2,m2) + " %20.12f"%val[i]+ " %20.12f"%val_prev[i])
                    i+=1
            
            #check for convergence
            diff = np.abs(val - val_prev)

            if (np.any(diff>quad_tol)):
                print(str(scheme)+" convergence not reached") 
                for i in range(len(val_prev)):
                    val_prev[i] = val[i]

            elif (np.all(diff<quad_tol)):     
                print(str(scheme)+" convergence reached!!!")
                exit()

            #if no convergence reached raise warning
            if (scheme == spherical_schemes[len(spherical_schemes)-1] and np.any(diff>quad_tol)):
                print("WARNING: convergence at tolerance level = " + str(quad_tol) + " not reached for all considered quadrature schemes")





if __name__ == "__main__":      


   # potmatrix.calc_potmatelem(l1=0,m1=0,l2=1,m2=1,0.0)

    """ Test angular convergence of the potential matrix elements """
    lmin = 0
    lmax = 4
    quad_tol = 1e-9

    potmatrix = potmat(scheme='lebedev_005')
    potmatrix.test_angular_convergence(lmin,lmax,quad_tol)