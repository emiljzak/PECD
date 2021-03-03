#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
import quadpy
from sympy.physics.wigner import gaunt
from sympy import *
from scipy.sparse import linalg
from scipy.special import sph_harm
from scipy import interpolate
from basis import angbas,radbas
from field import Field
import matplotlib.pyplot as plt
import time
#the imports below will have to be removed and calling of gauss lobatto should be made from another class
from sympy import symbols
from sympy.core import S, Dummy, pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
                                    hermite_poly, jacobi_poly)
from sympy.polys.rootoftools import RootOf
from sympy.core.compatibility import range

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

from multipoles import MultipoleExpansion


#print(quadpy.__version__)

class mapping():

    def __init__(self,lmin,lmax,nbins,nlobatto):
        self.lmin = lmin
        self.lmax = lmax
        self.nbins = nbins
        self.nlobatto = nlobatto


    def gen_map(self):
        imap=0
        maparray= []

        for l in range(0,self.lmax+1):
            for m in range(-l,l+1):
                for i in range(0,self.nbins):
                    for n in range (0,self.nlobatto):
                        if n==self.nlobatto-1:
                            continue        
                        elif n==0 and i==self.nbins-1:
                            continue
                        else:
                                imap+=1
                                #print(l,m,i,n,imap)
                                maparray.append([l,m,i,n,imap])
                
        fl = open('map.dat','w')
        for elem in maparray:
            #other formatting option:   	   	    
            #print("%5d %5d %5d %5d %5d"%[elem[i] for i in range(len(elem))]+"\n")

            #print(['{:5d}'.format(i) for i in elem])
 	   	    fl.write("%5d"%elem[0]+"  %5d"%elem[1]+ "  %5d"%elem[2]+  " %5d"%elem[3]+" %5d"%elem[4]+"\n")
        fl.close()
        
        return maparray, imap

class hmat():
    def __init__(self,params,t,rgrid,maparray):

        self.params = params
        self.t = t #time at which hamiltonian is evaluated
        self.rgrid = rgrid
        self.maparray = maparray
        self.rbas = radbas(params['nlobatto'], params['nbins'], params['binwidth'], params['rshift'])

    def calc_hmat(self):
        """ calculate static field-free Hamiltonian matrix """
        """ only called when using method = 'static','direct'"""

        hmat = np.zeros((self.params['Nbas'],self.params['Nbas'] ),dtype=complex)
        keomat = np.zeros((self.params['Nbas'],self.params['Nbas'] ),dtype=complex)
        potmat = np.zeros((self.params['Nbas'],self.params['Nbas'] ),dtype=complex)


        start_time = time.time()
        keomat = self.calc_keomat() 
        end_time = time.time()
        print("Time for construction of the KEO: " +  str("%10.3f"%(end_time-start_time)) + "s")
        #exit()
        start_time = time.time()
        potmat = self.calc_potmat()
        end_time = time.time()
        print("Time for construction of the potential energy matrix: " +  str("%10.3f"%(end_time-start_time)) + "s")


        hmat =   keomat + potmat

        neval = 10 #params["n"]
        print("Hamiltonian Matrix")
        with np.printoptions(precision=4, suppress=True, formatter={'complex': '{:15.8f}'.format}, linewidth=400):
            print(hmat)
        start_time = time.time()
        #eval, eigvec = linalg.eigs(-1.0 * hmat,k=neval,which='LM')
        eval, eigvec = np.linalg.eigh(hmat, UPLO='U')
        end_time = time.time()

        print("Time for diagonalization of field-free Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")

        print("ZPE = " + str(eval[0]))
        eval /=1.0
        print("Eigenvalues:"+'\n')
        """ Exact energies of hydrogen atom """
        evalhydrogen = np.zeros(neval,dtype=float)
        for i in range(1,neval+1):
            evalhydrogen[i-1] = - 1./(2.0 * float(i ** 2))

        evals = np.column_stack((eval[:neval],evalhydrogen))

        with np.printoptions(precision=4, suppress=True, formatter={'float': '{:12.5f}'.format}, linewidth=40):
            #print(eval-eval[0],evalhydrogen - evalhydrogen[0])
            print(evals)

        """figenr = plt.figure()
        plt.xlabel('levels')
        plt.ylabel('energy / a.u.')
        plt.legend()   
 
        plt.plot(np.linspace(0,1,np.size(eval[:10])), eval[:10]-eval[0],'bo-') 
        plt.plot(np.linspace(0,1,np.size(evalhydrogen[:10])), evalhydrogen[:10]-evalhydrogen[0],'ro-') 
        plt.show()     
        """
        #exit()
        #print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in hmat]))
        return evals, eigvec, hmat, keomat, potmat

    def calc_keomat(self):

        nlobatto = self.params['nlobatto']
        Nbas = self.params['Nbas']

        """call Gauss-Lobatto rule """
        x=np.zeros(nlobatto)
        w=np.zeros(nlobatto)
        x,w=self.gauss_lobatto(nlobatto,14)
        x=np.array(x)
        w=np.array(w)

        """calculate full keo matrix"""
        keomat = np.zeros((self.params['Nbas'] ,self.params['Nbas'] ), dtype = float)

        """ K(l1 m1 i1 n1,l1 m1 i1 n1) """ 

        for i in range(Nbas):
            rin = self.rgrid[self.maparray[i][2],self.maparray[i][3]]
            for j in range(i,Nbas):
                keomat[i,j] = self.calc_keomatel(self.maparray[i][0],self.maparray[i][2],self.maparray[i][3],self.maparray[j][2],self.maparray[j][3],x,w,rin)

        print("KEO matrix")
        with np.printoptions(precision=3, suppress=True, formatter={'float': '{:10.3f}'.format}, linewidth=400):
            print(0.5*keomat)
        return  0.5*keomat

    def calc_keomatel(self,l1,i1,n1,i2,n2,x,w,rin):
        "calculate matrix element of the KEO"

        if i1==i2 and n1==n2:
            KEO =  self.KEO_matel_rad(i1,n1,i2,n2,x,w) + self.KEO_matel_ang(i1,n1,l1,rin) 
            return KEO
        else:
            KEO = self.KEO_matel_rad(i1,n1,i2,n2,x,w)
            return KEO

                   
    def KEO_matel_rad(self,i1,n1,i2,n2,x,w):
        #w /= sqrt(sum(w[:]))
        w_i1 = w#/sum(w[:])
        w_i2 = w#/sum(w[:]) 

        nlobatto = self.params['nlobatto']
        if n1>0 and n2>0:
            if i1==i2:
                #single x single
                KEO = self.KEO_matel_fpfp(i1,n1,n2,x,w_i1) #checked
                return KEO/sqrt(w_i1[n1] * w_i2[n2])
            else:
                return 0.0
        if n1==0 and n2>0:
            #bridge x single
            if i1==i2: 
                KEO = self.KEO_matel_fpfp(i2,nlobatto-1,n2,x,w_i2) # not nlobatto -2?
                return KEO/sqrt(w_i2[n2]*(w_i1[nlobatto-1]+w_i1[0]))
            elif i1==i2-1:
                KEO=self.KEO_matel_fpfp(i2,0,n2,x,w_i2) # i2 checked  Double checked Feb 12
                return KEO/sqrt(w_i2[n2]*(w_i1[nlobatto-1]+w_i1[0]))
            else:
                return 0.0
        elif n1>0 and n2==0:
            #single x bridge
            if i1==i2: 
                KEO=self.KEO_matel_fpfp(i1,n1,nlobatto-1,x,w_i1) #check  Double checked Feb 12
                return KEO/sqrt(w_i1[n1]*(w_i2[nlobatto-1]+w_i2[0]))
            elif i1==i2+1:
                KEO=self.KEO_matel_fpfp(i1,n1,0,x,w_i1) #check  Double checked Feb 12
                return KEO/sqrt(w_i1[n1]*(w_i2[nlobatto-1]+w_i2[0]))
            else:
                return 0.0
                
        elif n1==0 and n2==0:
            #bridge x bridge
            if i1==i2: 
                KEO= ( self.KEO_matel_fpfp(i1,nlobatto-1,nlobatto-1,x,w_i1) + self.KEO_matel_fpfp(i1+1,0,0,x,w_i1) ) / sqrt((w_i1[nlobatto-1]+w_i1[0])*(w_i2[nlobatto-1]+w_i2[0])) #checked 10feb 2021   
                return KEO
            elif i1==i2-1:
                KEO = self.KEO_matel_fpfp(i2,nlobatto-1,0,x,w_i2) #checked 10feb 2021 Double checked Feb 12
                return KEO/sqrt((w_i1[nlobatto-1]+w_i1[0])*(w_i2[nlobatto-1]+w_i2[0]))
            elif i1==i2+1:
                KEO = self.KEO_matel_fpfp(i1,nlobatto-1,0,x,w_i1) #checked 10feb 2021. Double checked Feb 12
                return KEO/sqrt((w_i1[nlobatto-1]+w_i1[0])*(w_i2[nlobatto-1]+w_i2[0]))
            else:
                return 0.0

    def KEO_matel_ang(self,i1,n1,l,rgrid):
        """Calculate the anglar momentum part of the KEO"""
        """ we pass full grid and return an array on the full grid. If needed we can calculate only singlne element r_i,n """ 
        #r=0.5e00*(Rbin*x+Rbin*(i+1)+Rbin*i)+epsilon
        return float(l)*(float(l)+1)/((rgrid)**2)

    def KEO_matel_fpfp(self,i,n1,n2,x,w):
        "Calculate int_r_i^r_i+1 f'(r)f'(r) dr in the radial part of the KEO"
        # f'(r) functions from different bins are orthogonal
        #scale the Gauss-Lobatto points
        binwidth = self.params['binwidth']
        rshift = self.params['rshift']
        nlobatto = self.params['nlobatto']
        x_new = 0.5 * ( binwidth * x + binwidth * (i+1) + binwidth * i + rshift)  #checked
        #scale the G-L quadrature weights
        #w *= 0.5 * binwidth 

        fpfpint=0.0e00
        for k in range(0, nlobatto):
            y1 = self.rbas.fp(i,n1,k,x_new)#*sqrt(w[n1])
            y2 = self.rbas.fp(i,n2,k,x_new)#*sqrt(w[n2])
            fpfpint += w[k] * y1 * y2 #*0.5 * binwidth # 
    
        return fpfpint#sum(w[:])
        

    """ ======================= CONVERGENCE TESTS ======================"""    
    def test_angular_convergence(self,lmin,lmax,quad_tol,rin):

        #do decision tree and  optionally compare two results


        """ Choose the potential function to be used """
        potential_function = getattr(self, self.params['potential'] )
        if not potential_function:
            raise NotImplementedError("Method %s not implemented" %  self.params['potential'])
        print("potential function chosen: " + str(potential_function))

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
        #print("Available schemes: " + str(spherical_schemes))

        #iterate over the schemes
        for scheme in spherical_schemes[3:]: #skip 003a,003b,003c rules

            i=0
            for l1,m1 in anglist:
                for l2,m2 in anglist:
                
                    val[i] = self.calc_potmatelem(l1,m1,l2,m2,rin,scheme,potential_function)
                    #print(" %4d %4d %4d %4d"%(l1,m1,l2,m2) + " %20.12f"%val[i]+ " %20.12f"%val_prev[i])
                    i+=1
            
            #check for convergence
            diff = np.abs(val - val_prev)

            if (np.any(diff>quad_tol)):
                print(str(scheme)+" convergence not reached") 
                for i in range(len(val_prev)):
                    val_prev[i] = val[i]

            elif (np.all(diff<quad_tol)):     
                print(str(scheme)+" convergence reached!!!")
                break

            #if no convergence reached raise warning
            if (scheme == spherical_schemes[len(spherical_schemes)-1] and np.any(diff>quad_tol)):
                print("WARNING: convergence at tolerance level = " + str(quad_tol) + " not reached for all considered quadrature schemes")

    def test_angular_convergence_esp_interp(self,lmin,lmax,quad_tol,rin,esp):

        #create list of basis set indices
        anglist = []
        for l in range(lmin,lmax+1):
            for m in range(0,l+1):
                anglist.append([l,m])


        #convergence test over matrix elements:
        val =  np.zeros(shape=(len(anglist)**2),dtype=complex)
        val_prev = np.zeros(shape=(len(anglist)**2),dtype=complex)
        val_conv = np.zeros(shape=(len(anglist)**2),dtype=complex)
        if self.params['test_lebedev'] == True:
            print("Testing potential energy matrix elements using Lebedev quadratures")
            #pull out Lebedev schemes into a list
            spherical_schemes = []
            for elem in list(quadpy.u3.schemes.keys()):
                if 'lebedev' in elem:
                    spherical_schemes.append(elem)
            #print("Available schemes: " + str(spherical_schemes))

            #iterate over the schemes
            for scheme in spherical_schemes[3:]: #skip 003a,003b,003c rules

                i=0
                for l1,m1 in anglist:
                    for l2,m2 in anglist:
                    
                        val[i] = self.calc_potmatelem_esp(l1,m1,l2,m2,rin,scheme,esp)
                        print(" %4d %4d %4d %4d"%(l1,m1,l2,m2) + " %20.12f"%val[i]+ " %20.12f"%val_prev[i]+ " %20.12f"%(np.abs(val[i] - val_prev[i])))
                        i+=1
                
                #check for convergence
                diff = np.abs(val - val_prev)

                if (np.any(diff>quad_tol)):
                    print(str(scheme)+" convergence not reached") 
                    for i in range(len(val_prev)):
                        val_prev[i] = val[i]

                elif (np.all(diff<quad_tol)):     
                    print(str(scheme)+" convergence reached!!!")
                    val_conv = val_prev
                    break

                #if no convergence reached raise warning
                if (scheme == spherical_schemes[len(spherical_schemes)-1] and np.any(diff>quad_tol)):
                    print("WARNING: convergence at tolerance level = " + str(quad_tol) + " not reached for all considered quadrature schemes")

        if self.params['test_multipoles'] == True:
            print("Testing potential energy matrix elements using spherical harmonics expansion of the electrostatic potential")

            #convergence test over matrix elements:
            sphval =  np.zeros(shape=(len(anglist)**2),dtype=complex)
            sphval_prev = np.zeros(shape=(len(anglist)**2))
            sphval_conv = np.zeros(shape=(len(anglist)**2))
            VLM = np.zeros(shape=(self.params['multipoles_lmax']* (2*self.params['multipoles_lmax']+1)),dtype=complex)

            #build the matrix of spherical expansion coefficients
            i=0         
            myscheme = quadpy.u3.schemes["lebedev_019"]()                       
            for L in range(self.params['multipoles_lmax']+1):
                for M in range (-L,L+1):             
                    print("(L,M): " +str((L,M)))   
                    VLM[i] = myscheme.integrate_spherical(lambda theta_phi: np.conjugate(sph_harm(M, L,  theta_phi[1]+np.pi, theta_phi[0])) * self.pot_grid_psi4_d2s(esp,rin,theta_phi[0],theta_phi[1]+np.pi)) 
                    i+=1

            i=0
            for l1,m1 in anglist:
                for l2,m2 in anglist:
                    print(l1,m1,l2,m2)
                    j=0
                    for L in range(self.params['multipoles_lmax']+1):
                        for M in range (-L,L+1):
                            sphval[i] += VLM[j] * N(gaunt(l1,L,l2,m1,M,m2)) 
                            j+=1
                    print(" %4d %4d %4d %4d"%(l1,m1,l2,m2) + " %20.12f"%sphval[i])       
                    i+=1       
            
        print(sphval - val_conv)



    """ ======================= POTENTIALS ======================"""

    def pot_grid_interp(self):
        #interpolate potential on the grid
        fl = open(self.params['potential_grid'],'r')

        esp = []

        for line in fl:
            words = line.split()
            # columns: l, m, i, n, coef

            x = float(words[0])
            y = float(words[1])
            z = float(words[2])
            v = float(words[3])
            esp.append([x,y,z,v])
            
        esp = np.asarray(esp)

        #plot preparation
        X = np.linspace(min(esp[:,0]), max(esp[:,0]),100)
        Y = np.linspace(min(esp[:,1]), max(esp[:,1]),100)
        X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
        
        #fig = plt.figure() 
        #ax = plt.axes(projection="3d")
        #ax.scatter(x, z, v, c='r', marker='o')

        start_time = time.time()
        esp_interp = interpolate.LinearNDInterpolator(esp[:,0:3],esp[:,3])
        end_time = time.time()
        print("Interpolation of " + self.params['potential_grid'] + " potential took " +  str("%10.3f"%(end_time-start_time)) + "s")

        Z = esp_interp(X, Y, 0)
        plt.pcolormesh(X, Y, Z, shading='auto')
        #plt.plot(esp[:,0], esp[:,1], "o", label="input point")
        plt.colorbar()
        plt.axis("equal")
        plt.show()
        return esp_interp

    def pot_grid_interp_sph(self,interpolant,r,theta,phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return interpolant(x,y,z)

    def pot_ocs_pc(self,r,theta):
        #point charges from rigid OCS
        #partial charges in OCS

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

    def pot_model_d2s(self,r,theta,phi):

        rn = np.array([0.0, 2.5, 1.5])
        thetan = np.array([0.0, np.pi /2, 0.0])
        phin = np.array([0.0, np.pi /2, 0.0])
        q = np.array([-1.5, -0.75 , -0.75])
        Q = np.array([2.0, 1.0 , 1.0])

        Vne = 0.0
        for i in range(3):
            Vne += Q[i] / self.vecdist(r,theta,phi,rn[i],thetan[i],phin[i])


        Vee = 0.0
        for i in range(3):
            Vee += q[i] * self.vecdist(r,theta,phi,rn[i],thetan[i],phin[i])


        return Vne + Vee

    def vecdist(self,r,theta,phi,r0,theta0,phi0):
        return np.sqrt( r**2 + r0**2 - 2 * r * r0 * (np.sin(theta) * np.sin(theta0) * np.cos(phi - phi0) + np.cos(theta) * np.cos(theta0)) )

    def integrate_ee_pot(self,i,x,w):
        for k in range(len(x)):
            val_elem += w[k] * inv_weight_func(x[k]) * self.params['scheme'].integrate_spherical(lambda theta_phi: np.exp(- 1.0 * self.vecdist(r,theta,phi,r0,theta0,phi0) ) / self.vecdist(r,theta,phi,r0,theta0,phi0))

    def pot_chiral1(self,r,theta,phi):

        rn = np.array([0.0, 2.5, 1.5])
        thetan = np.array([0.0, np.pi /2, 0.0])
        phin = np.array([0.0, np.pi /2, 0.0])
        q = np.array([-1.5, -0.75 , -0.75])
        Q = np.array([2.0, 1.0 , 1.0])

        Vne = 0.0
        for i in range(3):
            Vne += Q[i] / self.vecdist(r,theta,phi,rn[i],thetan[i],phin[i])


        Vee = 0.0
        for i in range(3):
            Vee += q[i] * self.vecdist(r,theta,phi,rn[i],thetan[i],phin[i])


        return Vne + Vee



    def pot_test(self,r,theta,phi):
        d = 1.0
        """test potential for quadrature integration"""
        #d * np.cos(theta)**2 * np.cos(phi) / r**2
        return d * np.cos(theta) / r**2
  
    def pot_hydrogen(self,r,theta,phi):
        """test H-atom potential for quadrature integration"""
        #d * np.cos(theta)**2 * np.cos(phi) / r**2
        alpha = 0.005
        return  -1./np.sqrt(r**2+alpha**2)
        #return -1.0/r

    def pot_null(self,r,theta,phi):
        return 0.0 #zero-potential

    """ ======================= POTENTIALS ======================"""




    """ ======================= POTMAT ======================"""
    def calc_potmat(self):  
        """calculate full potential matrix"""
        lmax = self.params['lmax']
        lmin = self.params['lmin']
        Nbas = self.params['Nbas']
        scheme = self.params['scheme']
        #create list of basis set indices
        """anglist = []
        for l in range(lmin,lmax+1):
            for m in range(0,l+1):
                anglist.append([l,m])"""

        potmat = np.zeros((self.params['Nbas'] ,self.params['Nbas'] ), dtype = complex)


        if self.params['pot_type'] == "analytic":

            """ Choose the potential function to be used """
            potential_function = getattr(self, self.params['potential'] )
            if not potential_function:
                raise NotImplementedError("Method %s not implemented" %  self.params['potential'])
            print("potential function chosen: " + str(potential_function))
            

            """ V(l1 m1 i1 n1,l1 m1 i1 n1) """ 

            """calculate the <Y_l'm'(theta,phi)| V(r_in,theta,phi) | Y_lm(theta,phi)> integral """

            for i in range(Nbas):
                #ivec = [self.maparray[i][0],self.maparray[i][1],self.maparray[i][2],self.maparray[i][3]]
                #print(ivec)
                rin = self.rgrid[self.maparray[i][2],self.maparray[i][3]]

                for j in range(i,Nbas):
                    #jvec = [self.maparray[j][0],self.maparray[j][1],self.maparray[j][2],self.maparray[j][3]]
                    if self.maparray[i][2] == self.maparray[j][2] and self.maparray[i][3] == self.maparray[j][3]:
                        potmat[i,j] = self.calc_potmatelem(self.maparray[i][0],self.maparray[i][1],self.maparray[j][0],self.maparray[j][1],rin,scheme,potential_function)

            #print("Potential matrix")
            #with np.printoptions(precision=4, suppress=True, formatter={'complex': '{:15.8f}'.format}, linewidth=400):
            #    print(potmat)


        elif self.params['pot_type'] == "grid":  
            print("potential function chosen: " + str(self.params['potential_grid']))
            
            """ V(l1 m1 i1 n1,l1 m1 i1 n1) """ 
            """calculate the <Y_l'm'(theta,phi)| V(r_in,theta,phi) | Y_lm(theta,phi)> integral """

            #interpolate the ESP:
            print("Interpolating electrostatic potential")
            esp_interpolant = self.pot_grid_interp()

            for i in range(Nbas):
                #ivec = [self.maparray[i][0],self.maparray[i][1],self.maparray[i][2],self.maparray[i][3]]
                #print(ivec)
                rin = self.rgrid[self.maparray[i][2],self.maparray[i][3]]

                for j in range(i,Nbas):
                    #jvec = [self.maparray[j][0],self.maparray[j][1],self.maparray[j][2],self.maparray[j][3]]
                    if self.maparray[i][2] == self.maparray[j][2] and self.maparray[i][3] == self.maparray[j][3]:
                        potmat[i,j] = self.calc_potmatelem_interp(self.maparray[i][0],self.maparray[i][1],self.maparray[j][0],self.maparray[j][1],rin,scheme,esp_interpolant)

            print("Potential matrix")
            with np.printoptions(precision=4, suppress=True, formatter={'complex': '{:15.8f}'.format}, linewidth=400):
                print(potmat)
        return potmat

    def calc_potmatelem(self,l1,m1,l2,m2,rin,scheme,potential_function):
        """calculate single element of potential matrix"""
        myscheme = quadpy.u3.schemes[scheme]()
        #print(myscheme)
        """
        Symbols: 
                theta_phi[0] = theta in [0,pi]
                theta_phi[1] = phi  in [-pi,pi]
                sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) means that we put the phi angle in range [0,2pi] and the  theta angle in range [0,pi] as required by the scipy special funciton sph_harm
        """
        val = myscheme.integrate_spherical(lambda theta_phi: np.conjugate(sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) * sph_harm(m2, l2, theta_phi[1]+np.pi, theta_phi[0]) * potential_function(rin,theta_phi[0],theta_phi[1]+np.pi)) 

        return val
   

    def calc_potmatelem_interp(self,l1,m1,l2,m2,rin,scheme,interpolant):
        """calculate single element of potential matrix"""
        myscheme = quadpy.u3.schemes[scheme]()
        #print(myscheme)
        """
        Symbols: 
                theta_phi[0] = theta in [0,pi]
                theta_phi[1] = phi  in [-pi,pi]
                sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) means that we put the phi angle in range [0,2pi] and the  theta angle in range [0,pi] as required by the scipy special funciton sph_harm
        """
        val = myscheme.integrate_spherical(lambda theta_phi: np.conjugate(sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) * sph_harm(m2, l2, theta_phi[1]+np.pi, theta_phi[0]) * self.pot_grid_interp_sph(interpolant,rin,theta_phi[0],theta_phi[1]+np.pi)) 

        return val
   

    def test_potmat_accuracy(self,lmax_multipole,quad_tol,rin):

        """
        This method tests the convergence of potential energy matrix elements with options to use two methods:
        a) multipole expansion, with multipole moments calculated from library and analytic matrix elements
        b) Lebedev quadrature over the exact potential
        rin: radial coordinate
        """
        
        """ Choose the potential function to be used """
        potential_function = getattr(self, self.params['potential'] )
        if not potential_function:
            raise NotImplementedError("Method %s not implemented" %  self.params['potential'])
        print("potential function chosen: " + str(potential_function))

        #create list of basis set indices
        anglist = []
        for l in range(self.params['lmin'],self.params['lmax']+1):
            for m in range(0,l+1):
                anglist.append([l,m])


        if self.params['test_multipoles'] == True:
            print("calculating potential matrix elements using multipole expansion")
            #call multipoles

            pot_multipoles =  np.zeros(shape=(len(anglist)**2))
            pot_multipoles_temp = np.zeros(shape=(len(anglist)**2))



        if self.params['test_lebedev'] == True:
            print("calculating potential matrix elements using Lebedev quadratures")
            #call lebedev
            pot_lebedev =  np.zeros(shape=(len(anglist)**2))
            pot_lebedev_temp = np.zeros(shape=(len(anglist)**2))
            #pull out Lebedev schemes into a list
            spherical_schemes = []
            for elem in list(quadpy.u3.schemes.keys()):
                if 'lebedev' in elem:
                    spherical_schemes.append(elem)
            #print("Selected Lebedev schemes: " + str(spherical_schemes))

            #iterate over the schemes
            for scheme in spherical_schemes[3:]: #skip 003a,003b,003c rules

                i=0
                for l1,m1 in anglist:
                    for l2,m2 in anglist:
                    
                        pot_lebedev[i] = self.calc_potmatelem(l1,m1,l2,m2,rin,scheme,potential_function)
                        print(" %4d %4d %4d %4d"%(l1,m1,l2,m2) + " %20.12f"%pot_lebedev[i]+ " %20.12f"%pot_lebedev_temp[i]+ " %20.12f"%(pot_lebedev[i]-pot_lebedev_temp[i]))
                        i+=1
                
                #check for convergence
                diff = np.abs(pot_lebedev - pot_lebedev_temp)

                if (np.any(diff>quad_tol)):
                    print(str(scheme)+" convergence not reached") 
                    for i in range(len(pot_lebedev_temp)):
                        pot_lebedev_temp[i] = pot_lebedev[i]

                elif (np.all(diff<quad_tol)):     
                    print(str(scheme)+" convergence reached!!!")
                    break

                #if no convergence reached raise warning
                if (scheme == spherical_schemes[len(spherical_schemes)-1] and np.any(diff>quad_tol)):
                    print("WARNING: convergence at tolerance level = " + str(quad_tol) + " not reached for all considered quadrature schemes")

    def gauss_lobatto(self,n, n_digits):
            """
            Computes the Gauss-Lobatto quadrature [1]_ points and weights.

            The Gauss-Lobatto quadrature approximates the integral:

            .. math::
                \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

            The nodes `x_i` of an order `n` quadrature rule are the roots of `P'_(n-1)`
            and the weights `w_i` are given by:

            .. math::
                &w_i = \frac{2}{n(n-1) \left[P_{n-1}(x_i)\right]^2},\quad x\neq\pm 1\\
                &w_i = \frac{2}{n(n-1)},\quad x=\pm 1

            Parameters
            ==========

            n : the order of quadrature

            n_digits : number of significant digits of the points and weights to return

            Returns
            =======

            (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
                    The points `x_i` and weights `w_i` are returned as ``(x, w)``
                    tuple of lists.

            Examples
            ========

            >>> from sympy.integrals.quadrature import gauss_lobatto
            >>> x, w = gauss_lobatto(3, 5)
            >>> x
            [-1, 0, 1]
            >>> w
            [0.33333, 1.3333, 0.33333]
            >>> x, w = gauss_lobatto(4, 5)
            >>> x
            [-1, -0.44721, 0.44721, 1]
            >>> w
            [0.16667, 0.83333, 0.83333, 0.16667]

            See Also
            ========

            gauss_legendre,gauss_laguerre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi

            References
            ==========

            .. [1] https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules
            .. [2] http://people.math.sfu.ca/~cbm/aands/page_888.htm
            """
            x = Dummy("x")
            p = legendre_poly(n-1, x, polys=True)
            pd = p.diff(x)
            xi = []
            w = []
            for r in pd.real_roots():
                if isinstance(r, RootOf):
                    r = r.eval_rational(S(1)/10**(n_digits+2))
                xi.append(r.n(n_digits))
                w.append((2/(n*(n-1) * p.subs(x, r)**2)).n(n_digits))

            xi.insert(0, -1)
            xi.append(1)
            w.insert(0, (S(2)/(n*(n-1))).n(n_digits))
            w.append((S(2)/(n*(n-1))).n(n_digits))
            return xi, w

    def calc_intmat(self,time,intmat,field):  
        """calculate full interaction matrix"""
        Nbas = self.params['Nbas']

        #we keep the time variable if we wish to add analytic functions here

        #print("Electric field vector")
        #print(Fvec)
        #field: (E_1, E_-1, E_0) in spherical tensor form

        """ keep separate methods for cartesian and spherical tensor: to speed up by avoiding ifs"""

        """ Hint(l1 m1 i1 n1,l1 m1 i1 n1) """ 

        """calculate the <Y_l'm'(theta,phi)| d(theta,phi) | Y_lm(theta,phi)> integral """

        T = np.zeros(3)
        for i in range(Nbas):
            rin = self.rgrid[self.maparray[i][2],self.maparray[i][3]]

            for j in range(Nbas):
                if self.maparray[i][2] == self.maparray[j][2] and self.maparray[i][3] == self.maparray[j][3]:
                    T[0] = N(gaunt(self.maparray[i][0],1,self.maparray[j][0],self.maparray[i][1],1,self.maparray[j][1]))
                    T[1] = N(gaunt(self.maparray[i][0],1,self.maparray[j][0],self.maparray[i][1],-1,self.maparray[j][1]))
                    T[2] = N(gaunt(self.maparray[i][0],1,self.maparray[j][0],self.maparray[i][1],0,self.maparray[j][1]))


                    intmat[i,j] = np.sqrt(2.0 * np.pi / 3.0) * np.dot(field,T) * rin 

        intmat += np.conjugate(intmat.T)

        #print("Interaction matrix")
        #with np.printoptions(precision=4, suppress=True, formatter={'complex': '{:15.8f}'.format}, linewidth=400):
        #    print(intmat)

        return intmat

    def calc_intmatelem(self,l1,m1,i1,n1,l2,m2,i2,n2,scheme,field,rep_type,rin):
            """calculate single element of the interaction matrix"""
            myscheme = quadpy.u3.schemes[scheme]()
            #print(myscheme)
            """
            Symbols: 
                    theta_phi[0] = theta in [0,pi]
                    theta_phi[1] = phi  in [-pi,pi]
                    sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) means that we put the phi angle in range [0,2pi] and the  theta angle in range [0,pi] as required by the scipy special funciton sph_harm
            """

            """ int_rep_type: cartesian or spherical"""
            T = np.zeros(3)
            if rep_type == 'cartesian':
                T[0] = myscheme.integrate_spherical(lambda theta_phi: np.conjugate(sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) * sph_harm(m2, l2, theta_phi[1]+np.pi, theta_phi[0]) * np.sin(theta_phi[0]) * np.cos(theta_phi[1]+np.pi) )
                T[1] = myscheme.integrate_spherical(lambda theta_phi: np.conjugate(sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) * sph_harm(m2, l2, theta_phi[1]+np.pi, theta_phi[0]) * np.sin(theta_phi[0]) * np.sin(theta_phi[1]+np.pi) )
                T[2] = myscheme.integrate_spherical(lambda theta_phi: np.conjugate(sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) * sph_harm(m2, l2, theta_phi[1]+np.pi, theta_phi[0]) * np.cos(theta_phi[0])  )
                val = np.dot(field,T)  

            elif rep_type == 'spherical':

                T[0] = N(gaunt(l1,1,l2,m1,1,m2))
                T[1] = N(gaunt(l1,1,l2,m1,-1,m2))
                T[2] = N(gaunt(l1,1,l2,m1,0,m2)) * np.sqrt(2.0) #spherical tensor rank-1 coefficient
                val =  np.sqrt(2.0 * np.pi / 3.0) * np.dot(field,T) * rin
                val += np.conjugate( np.sqrt(2.0 * np.pi / 3.0)  * np.dot(field,T) * rin )
            return val/2.0 #?


def gaussian(XYZ, xyz0, sigma):
    g = np.ones_like(XYZ[0])
    for k in range(3):
        g *= np.exp(-(XYZ[k] - xyz0[k])**2 / sigma**2)
    g *= (sigma**2*np.pi)**-1.5
    return g


def test_multipoles():
    # First we set up our grid, a cube of length 10 centered at the origin:

    npoints = 101
    edge = 10
    x, y, z = [np.linspace(-edge/2., edge/2., npoints)]*3
    XYZ = np.meshgrid(x, y, z, indexing='ij')
    d = 1.0

    # We model our smeared out charges as gaussian functions:

    sigma = 1.5   # the width of our gaussians

    # Initialize the charge density rho, which is a 3D numpy array:
    rho = gaussian(XYZ, (0, 0, d), sigma) 


    # Prepare the charge distribution dict for the MultipoleExpansion object:

    charge_dist = {
        'discrete': False,     # we have a continuous charge distribution here
        'rho': rho,
        'xyz': XYZ
    }

    # The rest is the same as for the discrete case:

    l_max = 2   # where to stop the infinite multipole sum; here we expand up to the quadrupole (l=2)

    Phi = MultipoleExpansion(charge_dist, l_max)

    x, y, z = 0.0, 0.0, 0.9
    value = Phi(x, y, z)
    print(value)

    # The multipole moments are stored in a dict, where the keys are (l, m) and the values q_lm:
    for l in range(0,l_max +1):
        for m in range(-l,l+1):
            print(l,m, Phi.multipole_moments[(l,m)])


def test_multipoles_discr():
    # First we set up our grid, a cube of length 10 centered at the origin:

    npoints = 11
    edge = 10
    x, y, z = [np.linspace(-edge/2., edge/2., npoints)] * 3
    print(x,y,z)

    XYZ = np.meshgrid(x, y, z, indexing='ij')


    # Prepare the charge distribution dict for the MultipoleExpansion object:
    d = 1.0
    q = [0.1, 0.2, 0.25, 0.3, 0.15] 
    charge_dist = {
    'discrete': True,     # we have a continuous charge distribution here
    'charges': [{'q': q[0], 'xyz': (0, 0, 0)}, {'q': q[1], 'xyz': (0, 0, 0.9 * d)}, {'q': q[2], 'xyz': (1.2 * d, 0, 0)}, {'q': q[3], 'xyz': (0, d, 0)}, {'q': q[4], 'xyz': (-d/2., -d/2., -d/2.)}]}
    # The rest is the same as for the discrete case:

    l_max = 10   # where to stop the infinite multipole sum; here we expand up to the quadrupole (l=2)

    Phi = MultipoleExpansion(charge_dist, l_max)

    x, y, z = 4.0, 0.0, 3.0
    value = Phi(x, y, z)
    print(value)

    # The multipole moments are stored in a dict, where the keys are (l, m) and the values q_lm:
    for l in range(0,l_max +1):
        for m in range(-l,l+1):
            print(l,m, Phi.multipole_moments[(l,m)])


def rho(XYZ,rn,thetan,phin):
    g = np.ones_like(XYZ[0])
    g *= np.exp(-XYZ[0]**2)#np.exp( - 1.0 * np.sqrt( XYZ[0]**2 + XYZ[1]**2 + rn[1]**2 + 2.0 * np.sqrt(XYZ[0]**2 + XYZ[1]**2 + XYZ[2]**2) * rn[1] * XYZ[1] ) ) 
    #print(  np.sqrt( XYZ[0]**2 + XYZ[1]**2 + rn[1]**2 + 2.0 * np.sqrt(XYZ[0]**2 + XYZ[1]**2 + XYZ[2]**2) * rn[1] * XYZ[1] ) ) 
    return g


def plot_pot_chiral1():

    phi = np.pi / 2.0 #YZ plane
    rn = np.array([0.0, 2.5, 1.5])
    thetan = np.array([0.0, np.pi /2, 0.0])
    phin = np.array([0.0, np.pi /2, 0.0])
    q = np.array([-1.5, -0.75 , -0.75])
    Q = np.array([2.0, 1.0 , 1.0])

    """plot the angular basis"""
    theta_1d = np.linspace(0,   np.pi,  2*91) # 
    phi_1d   = np.linspace(0, 2*np.pi, 2*181) # 

    theta_2d, phi_2d = np.meshgrid(theta_1d, phi_1d)
    xyz_2d = np.array([np.sin(theta_2d) * np.sin(phi_2d), np.sin(theta_2d) * np.cos(phi_2d), np.cos(theta_2d)]) #2D grid of cartesian coordinates

    colormap = cm.ScalarMappable( cmap=plt.get_cmap("cool") )
    colormap.set_clim(-.45, .45)


    plt.figure()
    ax = plt.gca(projection = "3d")
    
    plt.title("Effective electrostatic potential")

    x = np.linspace(-10.0, 10.0 ,10, True, dtype=float)
    y = np.linspace(-10.0, 10.0 ,10, True, dtype=float)
    z = np.linspace(-10.0, 10.0 ,10, True, dtype=float)
    XYZ = np.meshgrid( z, y, x, indexing='ij')

    # Initialize the charge density rho, which is a 3D numpy array:
    #rho = lambda r, theta, phi: np.exp( - 1.0 * np.sqrt(r**2 + rn[1]**2 - 2.0 * r * rn[1] * np.sin(phi) * np.sin(theta) ) ) #gaussian(XYZ, (0, 0, 1), sigma) + gaussian(XYZ, (0, 0, 0), sigma) +  gaussian(XYZ, (0, 1, 0), sigma)

    #rho = lambda x, y, z: np.exp( - 1.0 * np.sqrt(x**2+y**2 + rn[1]**2 - 2.0 * np.sqrt(x**2 + y**2 + z**2) * rn[1] *y ) ) #gaussian(XYZ, (0, 0, 1), sigma) + gaussian(XYZ, (0, 0, 0), sigma) +  gaussian(XYZ, (0, 1, 0), sigma)

    dens = rho(XYZ,rn,thetan,phin)
    print(np.shape(XYZ))
    print(np.shape(dens))
    exit()
    # Prepare the charge distribution dict for the MultipoleExpansion object:

    charge_dist = {
        'discrete': False,     # we have a continuous charge distribution here
        'rho': dens,
        'xyz': XYZ
    }


    # The rest is the same as for the discrete case:

    l_max = 2   # where to stop the infinite multipole sum; here we expand up to the quadrupole (l=2)

    Vne = MultipoleExpansion(charge_dist, l_max)
    #print(Vne(5.0, 2.0, 1.0))
    #print(Vne.multipole_moments)
    print(type(Vne))
    exit()
    ax.plot_surface(XYZ[0], XYZ[1], Vne(XYZ[0],XYZ[1],XYZ[2]))

    #ax.set_aspect("equal")
    #ax.set_axis_off()
    
            
    plt.show()






#test_multipoles_discr()
#test_multipoles()
#plot_pot_chiral1()