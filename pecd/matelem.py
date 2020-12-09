#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
from basis import angbas
import quadpy
from scipy.special import sph_harm
from basis import radbas
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

        for l in range(0,lmax+1):
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

            print(['{:5d}'.format(i) for i in elem])
 	   	    #fl.write("%5d"%elem[0]+"  %5d"%elem[1]+"  %5d"%elem[2]+" %5d"%elem[3]+" %5d"%elem[4]+"\n")
        fl.close()
        
        return imap

class potential():
    """Class containing methods for the representation of the PES"""

    def __init__(self):
        pass

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

    def pot_test(self,r,theta,phi):
        d = 1.0
        """test potential for quadrature integration"""
        #d * np.cos(theta)**2 * np.cos(phi) / r**2
        return d * np.cos(theta) / r**2

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

        Nbas = int(len(anglist) * len(rin))
        vmat = np.zeros((Nbas,Nbas), dtype = float)

        """calculate the <Y_l'm'(theta,phi)| V(r_in,theta,phi) | Y_lm(theta,phi)> integral """
        for i in range(Nbas):
            ivec = [indmap[i][0],indmap[i][1],indmap[i][2]]
            print(type(ivec))
            print(ivec)
            for j in range(Nbas):
                jvec = [indmap[j][0],indmap[j][1],indmap[j][2]]
                #hmat[ivec,jvec] = self.helem(ivec,jvec,grid_dp,weights_dp)
                vmat[i,j] = self.calc_potmatelem(l1,m1,l2,m2,rin,self.scheme)


        print(vmat)
        eval, eigvec = np.linalg.eigh(hmat)
        print(eval)
        return eval
        

    def calc_potmatelem(self,l1,m1,l2,m2,rin,scheme):
        """calculate single element of potential matrix"""
        myscheme = quadpy.u3.schemes[scheme]()
        #print(myscheme)
        """
        Symbols: 
                theta_phi[0] = theta in [0,pi]
                theta_phi[1] = phi  in [-pi,pi]
                sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) means that we put the phi angle in range [0,2pi] and the  theta angle in range [0,pi] as required by the scipy special funciton sph_harm
        """
        val = myscheme.integrate_spherical(lambda theta_phi: np.conjugate(sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) * sph_harm(m2, l2, theta_phi[1]+np.pi, theta_phi[0]) * self.pot_ocs_pc(rin,theta_phi[0])) #* self.pot_test(rin, theta_phi[0], theta_phi[1]+np.pi) )

        return val
    
    def test_angular_convergence(self,lmin,lmax,quad_tol,rin):
        """
        rin: radial coordinate
        """
        
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
                
                    val[i] = self.calc_potmatelem(l1,m1,l2,m2,rin,scheme)
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

class keomat(radbas):
    """Class containing methods for the calculation of the KEO matrix elements"""

    def __init__(self):
        pass

    def calc_mat(self,Ntotal):
        "we define the KEO matrix through mapping into 2-D array"
        nlobatto=globals.nlobatto
        "initialize the KEO matrix:"
        KEO=np.zeros(shape=(Ntotal,Ntotal), dtype=float)
        "print(KEO[0][5])"
        "print(KEO.shape)"
        
        """call Gauss-Lobatto rule """
        x=np.zeros(nlobatto)
        w=np.zeros(nlobatto)
        x,w=gauss_lobatto(nlobatto,14)
        x=np.array(x)
        w=np.array(w)

        
        "Read the mapping table"
        maparray=np.ndarray(shape=(Ntotal,5), dtype=int)
        with open("map.txt", "r") as mapfile:                  
            maparray=np.loadtxt(mapfile,dtype=int)
        #print(maparray)
        """print(maparray[3][2],maparray[3][3],maparray[3][0],maparray[3][1],
                maparray[2][2],maparray[2][3],maparray[2][0],maparray[2][1])"""
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        for alpha in range(0,Ntotal):
            for beta in range(0,Ntotal):
                KEO[alpha][beta] = 1.0 * self.calc_keomatel(maparray[alpha][0],maparray[alpha][1],maparray[alpha][2],maparray[alpha][3],\
                maparray[beta][0],maparray[beta][1],maparray[beta][2],maparray[beta][3],x,w)
            "print(KEO[alpha])"
        
        with open("KEO.txt", "wb") as KEOfile:   #Pickling                
            np.savetxt(KEOfile,KEO,fmt='%10.3f')
        return KEO

    def calc_keomatel(self,l1,m1,i1,n1,l2,m2,i2,n2,x,w):
        "calculate matrix element of the KEO"
        
        if l1==l2 and m1==m2:
            
            if i1==i2 and n1==n2:
                KEO = self.KEO_matel_ang(i1,n1,l1,x) + self.KEO_matel_rad(i1,n1,i2,n2,x,w)
                return KEO
            else:
                KEO = self.KEO_matel_rad(i1,n1,i2,n2,x,w)
                return KEO
        else:
            return 0.0
                   
    def KEO_matel_rad(self,i1,n1,i2,n2,x,w):
        
        if n1>0 and n2>0:
            if i1==i2:
                #single x single
                KEO=0.5*self.KEO_matel_fpfp(i1,n1,n2,x,w_i1) 
                return KEO/sqrt(w_i1[n1]*w_i2[n2])
            else:
                return 0.0
        if n1==0 and n2>0:
            #bridge x single
            if i1==i2: 
                KEO=0.5*KEO_matel_fpfp(i1,nlobatto-1,n2,x,w_i2) 
                return KEO/sqrt(w_i2[n2]*(w_i1[nlobatto-1]+w_i1[0]))
            elif i1==i2-1:
                KEO=0.5*KEO_matel_fpfp(i2,0,n2,x,w_i2) 
                return KEO/sqrt(w_i2[n2]*(w_i1[nlobatto-1]+w_i1[0]))
            else:
                return 0.0
        elif n2==0 and n1>0:
            #single x bridge
            if i1==i2: 
                KEO=0.5*KEO_matel_fpfp(i1,n1,nlobatto-1,x,w_i1) 
                return KEO/sqrt(w_i1[n1]*(w_i2[nlobatto-1]+w_i2[0]))
            elif i1==i2+1:
                KEO=0.5*KEO_matel_fpfp(i1,n1,0,x,w_i1) 
                return KEO/sqrt(w_i1[n1]*(w_i2[nlobatto-1]+w_i2[0]))
            else:
                return 0.0
                
        elif n1==0 and n2==0:
            #bridge x bridge
            if i1==i2: 
                KEO=0.5*(KEO_matel_fpfp(i1,nlobatto-1,nlobatto-1,x,w_i1)+KEO_matel_fpfp(i1+1,0,0,x,w_i1))
                #print("shape of w is", np.shape(KEO), "and type of w is", type(w_i1))
            
                return KEO/sqrt((w_i1[nlobatto-1]+w_i1[0])*(w_i2[nlobatto-1]+w_i2[0]))
            elif i1==i2-1:
                KEO=0.5*KEO_matel_fpfp(i2,0,nlobatto-1,x,w_i2)
                return KEO/sqrt((w_i1[nlobatto-1]+w_i1[0])*(w_i2[nlobatto-1]+w_i2[0]))
            elif i1==i2+1:
                KEO=0.5*KEO_matel_fpfp(i1,0,nlobatto-1,x,w_i1)
                return KEO/sqrt((w_i1[nlobatto-1]+w_i1[0])*(w_i2[nlobatto-1]+w_i2[0]))
            else:
                return 0.0

    def KEO_matel_fpfp(self,i,n1,n2,x,w):
        "Calculate int_r_i^r_i+1 f'(r)f'(r) dr in the radial part of the KEO"
        # f'(r) functions from different bins are orthogonal
        #scale the Gauss-Lobatto points
        x = 0.5e00 * ( self.binwidth * x + self.binwidth * (i+1) + self.binwidth * i ) + self.rshift
        #scale the G-L quadrature weights
        w = 0.5 * self.binwidth * w

        fpfpint=0.0e00
        for k in range(0, self.nlobatto):
            y1 = self.fp(i,n1,k,x)
            y2 = self.fp(i,n2,k,x)
            fpfpint += w[k] * y1 * y2
    
        return float(fpfpint)
        

    def KEO_matel_ang(self,l,rgrid):
        """Calculate the anglar momentum part of the KEO"""
        """ we pass full grid and return an array on the full grid. If needed we can calculate only singlne element r_i,n """ 
        #r=0.5e00*(Rbin*x+Rbin*(i+1)+Rbin*i)+epsilon
        return float(l)*(float(l)+1)/(2.0*(rgrid)**2)

class hmat():
    def __init__(self,potential,field,params,t):
        self.potential = potential
        self.params = params
        self.t = t #time at which hamiltonian is evaluated
        self.field = field #field file

    def calc_mat(self):
        """ calculate full Hamiltonian matrix """
        """ only called when using method = 'static','direct'"""


        return hmat

if __name__ == "__main__":      

    lmin = 0
    lmax = 2
    # potmatrix.calc_potmatelem(l1=0,m1=0,l2=1,m2=1,0.0)

    """ Test angular convergence of the potential matrix elements """

    """ quad_tol = 1e-6

    potmatrix = potmat(scheme='lebedev_005')
    rin = np.linspace(0.01,100.0,10,endpoint=True)
    print(rin)
    for r in rin:
        print("radial coordinate = "+str(r))
        potmatrix.test_angular_convergence(lmin,lmax,quad_tol,r)
    """

    """ Test print the KEO matrix """
    """keomatrix = keomat()
    keomatrix.calc_mat"""

    """ Generate map """
    mymap = mapping(lmin = 0, lmax = 2, nbins = 3, nlobatto=4)
    mymap.gen_map()