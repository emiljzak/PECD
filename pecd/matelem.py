#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
import quadpy
from scipy.special import sph_harm
from basis import angbas,radbas
from field import Field
import matplotlib.pyplot as plt

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
                        #elif n==0 and i==self.nbins-1:
                        #    continue
                        else:
                                imap+=1
                                print(l,m,i,n,imap)
                                maparray.append([l,m,i,n,imap])
                
        fl = open('map.dat','w')
        for elem in maparray:
            #other formatting option:   	   	    
            #print("%5d %5d %5d %5d %5d"%[elem[i] for i in range(len(elem))]+"\n")

            #print(['{:5d}'.format(i) for i in elem])
 	   	    fl.write("%5d"%elem[0]+"  %5d"%elem[1]+"  %5d"%elem[2]+" %5d"%elem[3]+" %5d"%elem[4]+"\n")
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
        """ calculate full Hamiltonian matrix """
        """ only called when using method = 'static','direct'"""

        #print(self.params['lmin'], self.params['lmax'], self.params['nbins'], self.params['nlobatto'])
    
        hmat = np.zeros((self.params['Nbas'],self.params['Nbas'] ),dtype=float)

        hmat = self.calc_potmat() + self.calc_keomat() + self.calc_intmat()

        print("Hamiltonian Matrix")
        with np.printoptions(precision=4, suppress=True, formatter={'float': '{:15.8f}'.format}, linewidth=400):
            print(hmat)
        print("Eigenvalues"+'\n')
        eval, eigvec = np.linalg.eigh(hmat,UPLO='U')

        eval /=1.0


        evalhydrogen = np.zeros((self.params['Nbas']),dtype=float)
        for i in range(1,self.params['Nbas']+1):
            evalhydrogen[i-1] = - 1./(2.0 * float(i **2))

        #evals = np.column_stack((eval-eval[0],evalhydrogen-evalhydrogen[0]))
        evals = np.column_stack((eval,evalhydrogen))

        with np.printoptions(precision=4, suppress=True, formatter={'float': '{:15.8f}'.format}, linewidth=40):
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
        return evals, eigvec, hmat



    def calc_keomat(self):

        nlobatto = self.params['nlobatto']
        Nbas = self.params['Nbas']

        """call Gauss-Lobatto rule """
        x=np.zeros(nlobatto)
        w=np.zeros(nlobatto)
        x,w=self.gauss_lobatto(nlobatto,14)
        x=np.array(x)
        w=np.array(w)
        print("sum of weights")
        print(sum(w[:]))



        """calculate full keo matrix"""
        keomat = np.zeros((self.params['Nbas'] ,self.params['Nbas'] ), dtype = float)

        """ K(l1 m1 i1 n1,l1 m1 i1 n1) """ 

        for i in range(Nbas):
            rin = self.rgrid[self.maparray[i][2],self.maparray[i][3]]
            for j in range(Nbas):
                keomat[i,j] = self.calc_keomatel(self.maparray[i][0],self.maparray[i][1],self.maparray[i][2],self.maparray[i][3],self.maparray[j][0],self.maparray[j][1],self.maparray[j][2],self.maparray[j][3],x,w,rin)

        print("KEO matrix")
        #with np.printoptions(precision=4, suppress=True, formatter={'float': '{:15.8f}'.format}, linewidth=400):
        #    print(keomat)
        return  keomat

    def calc_keomatel(self,l1,m1,i1,n1,l2,m2,i2,n2,x,w,rin):
        "calculate matrix element of the KEO"
        
        if l1==l2 and m1==m2:
            
            if i1==i2 and n1==n2:
                KEO =  self.KEO_matel_rad(i1,n1,i2,n2,x,w) + self.KEO_matel_ang(i1,n1,l1,rin) 
                return KEO
            else:
                KEO = self.KEO_matel_rad(i1,n1,i2,n2,x,w)
                return KEO
        else:
            return 0.0
                   
    def KEO_matel_rad(self,i1,n1,i2,n2,x,w):
        w_i1 = w/sum(w[:])
        w_i2 = w/sum(w[:])
        nlobatto = self.params['nlobatto']
        if n1>0 and n2>0:
            if i1==i2:
                #single x single
                KEO=0.5*self.KEO_matel_fpfp(i1,n1,n2,x,w_i1) #checked
                return KEO/sqrt(w_i1[n1]*w_i2[n2])
            else:
                return 0.0
        if n1==0 and n2>0:
            #bridge x single
            if i1==i2: 
                KEO=0.5*self.KEO_matel_fpfp(i2,nlobatto-1,n2,x,w_i2) #checked
                return KEO/sqrt(w_i2[n2]*(w_i1[nlobatto-1]+w_i1[0]))
            elif i1==i2-1:
                KEO=0.5*self.KEO_matel_fpfp(i2,0,n2,x,w_i2) # i2 or i1?
                return KEO/sqrt(w_i2[n2]*(w_i1[nlobatto-1]+w_i1[0]))
            else:
                return 0.0
        elif n1>0 and n2==0:
            #single x bridge
            if i1==i2: 
                KEO=0.5*self.KEO_matel_fpfp(i1,n1,nlobatto-1,x,w_i1) #check
                return KEO/sqrt(w_i1[n1]*(w_i2[nlobatto-1]+w_i2[0]))
            elif i1==i2+1:
                KEO=0.5*self.KEO_matel_fpfp(i1,n1,0,x,w_i1) #check
                return KEO/sqrt(w_i1[n1]*(w_i2[nlobatto-1]+w_i2[0]))
            else:
                return 0.0
                
        elif n1==0 and n2==0:
            #bridge x bridge
            if i1==i2: 
                KEO=0.5*(self.KEO_matel_fpfp(i1,nlobatto-1,nlobatto-1,x,w_i1)+self.KEO_matel_fpfp(i1+1,0,0,x,w_i1))   #check         
                return KEO/sqrt((w_i1[nlobatto-1]+w_i1[0])*(w_i2[nlobatto-1]+w_i2[0]))
            elif i1==i2-1:
                KEO=0.5*self.KEO_matel_fpfp(i2,0,nlobatto-1,x,w_i2) #check
                return KEO/sqrt((w_i1[nlobatto-1]+w_i1[0])*(w_i2[nlobatto-1]+w_i2[0]))
            elif i1==i2+1:
                KEO=0.5*self.KEO_matel_fpfp(i1,0,nlobatto-1,x,w_i1)
                return KEO/sqrt((w_i1[nlobatto-1]+w_i1[0])*(w_i2[nlobatto-1]+w_i2[0]))
            else:
                return 0.0

    def KEO_matel_fpfp(self,i,n1,n2,x,w):
        "Calculate int_r_i^r_i+1 f'(r)f'(r) dr in the radial part of the KEO"
        # f'(r) functions from different bins are orthogonal
        #scale the Gauss-Lobatto points
        binwidth = self.params['binwidth']
        rshift = self.params['rshift']
        nlobatto = self.params['nlobatto']
        x_new = 0.5e00 * ( binwidth * x + binwidth * (i+1) + binwidth * i ) + rshift #checked
        #scale the G-L quadrature weights
        w = 0.5 * binwidth * w

        fpfpint=0.0e00
        for k in range(0, nlobatto):
            y1 = self.rbas.fp(i,n1,k,x_new)
            y2 = self.rbas.fp(i,n2,k,x_new)
            fpfpint += w[k] * y1 * y2/sum(w[:])# * x_new[k]**2
    
        return fpfpint
        
    def KEO_matel_ang(self,i1,n1,l,rgrid):
        """Calculate the anglar momentum part of the KEO"""
        """ we pass full grid and return an array on the full grid. If needed we can calculate only singlne element r_i,n """ 
        #r=0.5e00*(Rbin*x+Rbin*(i+1)+Rbin*i)+epsilon
        return float(l)*(float(l)+1)/(2.0*(rgrid)**2)

 
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
  
    def pot_hydrogen(self,r,theta,phi):
        """test H-atom potential for quadrature integration"""
        #d * np.cos(theta)**2 * np.cos(phi) / r**2
        return -1./r

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

        potmat = np.zeros((self.params['Nbas'] ,self.params['Nbas'] ), dtype = float)

        """ V(l1 m1 i1 n1,l1 m1 i1 n1) """ 

        """calculate the <Y_l'm'(theta,phi)| V(r_in,theta,phi) | Y_lm(theta,phi)> integral """

        for i in range(Nbas):
            ivec = [self.maparray[i][0],self.maparray[i][1],self.maparray[i][2],self.maparray[i][3]]
            #print(ivec)
            rin = self.rgrid[self.maparray[i][2],self.maparray[i][3]]
            for j in range(Nbas):
                jvec = [self.maparray[j][0],self.maparray[j][1],self.maparray[j][2],self.maparray[j][3]]
                if self.maparray[i][2] == self.maparray[j][2] and self.maparray[i][3] == self.maparray[j][3]:
                    potmat[i,j] = self.calc_potmatelem(self.maparray[i][0],self.maparray[i][1],self.maparray[j][0],self.maparray[j][1],rin,scheme)

        print("Potential matrix")
        #with np.printoptions(precision=4, suppress=True, formatter={'float': '{:15.8f}'.format}, linewidth=400):
        #    print(potmat)

        return potmat
        
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
        val = myscheme.integrate_spherical(lambda theta_phi: np.conjugate(sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) * sph_harm(m2, l2, theta_phi[1]+np.pi, theta_phi[0]) * self.pot_hydrogen(rin,theta_phi[0],theta_phi[1]+np.pi)) #* rin**2 #self.pot_ocs_pc(rin,theta_phi[0])) #* self.pot_test(rin, theta_phi[0], theta_phi[1]+np.pi) )

        return val
   
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



    def calc_intmat(self):  
        """calculate full interaction matrix"""
        lmax = self.params['lmax']
        lmin = self.params['lmin']
        Nbas = self.params['Nbas']
        scheme = self.params['scheme']
        int_rep_type = self.params['int_rep_type']

        """Load electric field at time t """
        Elfield = Field(self.params)
        Fvec = Elfield.gen_field()
        print("Electric field vector")
        print(Fvec)

        intmat = np.zeros((self.params['Nbas'] ,self.params['Nbas'] ), dtype = float)

        """ Hint(l1 m1 i1 n1,l1 m1 i1 n1) """ 

        """calculate the <Y_l'm'(theta,phi)| d(theta,phi) | Y_lm(theta,phi)> integral """

        for i in range(Nbas):
            rin = self.rgrid[self.maparray[i][2],self.maparray[i][3]]
            for j in range(i,Nbas):
                if self.maparray[i][2] == self.maparray[j][2] and self.maparray[i][3] == self.maparray[j][3]:
                    intmat[i,j] = rin * self.calc_intmatelem(self.maparray[i][0],self.maparray[i][1],self.maparray[j][0],self.maparray[j][1],scheme,Fvec,int_rep_type)

        print("Interaction matrix")
        with np.printoptions(precision=4, suppress=True, formatter={'float': '{:15.8f}'.format}, linewidth=400):
            print(intmat)

        return intmat


    def calc_intmatelem(self,l1,m1,l2,m2,scheme,field,rep_type):
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
                return 0
  
            return val

