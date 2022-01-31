#!/usr/bin/env python
# -*- coding: utf-8; fill-column: 120 -*-
#
import numpy as np

class Field():
    """Class representing electric field

    Args:
        filename : str
            Name of the HDF5 file from which tensor data is loaded.
    Kwargs:
        thresh : float
            Threshold for neglecting matrix elements when reading from file

    Attrs:
        rank : int
            Rank of tensor operator.
    """

    def __init__(self,params):
        self.params = params

    """ =================================== FIELD TYPES =================================== """
    def fieldRCPL(self, t, function_name, omega, E0, CEP0, spherical):
        #all vectors are returned in spherical tensor form -1, 0, 1 order. Adopted convention used in Artemyev et al., J. Chem. Phys. 142, 244105 (2015).
        if spherical == True:
            fieldvec = E0 * np.array( [  -1.0 * np.cos( omega * t + CEP0 )  + 1j * np.sin( omega * t + CEP0 ) , 0.0,  np.cos( omega * t + CEP0 )  + 1j * np.sin( omega * t + CEP0 )  ] ) / np.sqrt(2.0)
        else:
            fieldvec = E0 * np.array( [  -1.0 * np.cos( omega * t + CEP0 )  + 1j * np.sin( omega * t + CEP0 ) , 0.0,  np.cos( omega * t + CEP0 )  + 1j * np.sin( omega * t + CEP0 )  ] ) / np.sqrt(2.0)        
        return fieldvec 


    def fieldLCPL(self, t, function_name, omega, E0, CEP0, spherical):
        #all vectors are returned in spherical tensor form -1, 0, 1 order
        if spherical == True:
            fieldvec = E0 * np.array( [    np.cos( omega * t + CEP0 )  + 1j * np.sin( omega * t + CEP0 ) , 0.0,   -1.0 *np.cos( omega * t + CEP0 )  + 1j * np.sin( omega * t + CEP0 ) ] ) / np.sqrt(2.0)
        else:
            fieldvec = E0 * np.array( [    np.cos( omega * t + CEP0 )  + 1j * np.sin( omega * t + CEP0 ) , 0.0,   -1.0 *np.cos( omega * t + CEP0 )  + 1j * np.sin( omega * t + CEP0 ) ] ) / np.sqrt(2.0)
        return fieldvec 

    def fieldLP(self, t, function_name, omega, E0, CEP0):
        #fieldvec = t
        #print("shapes in fieldLP")
        #print(np.shape(t))
        #print(np.shape(np.cos( omega * t + CEP0 )))
        return 0.0, E0 * np.cos( omega * t + CEP0 ), 0.0


    """ =================================== FIELD ENVELOPES =================================== """
    def envgaussian(self,t,function_name,FWHM,t0):
        fieldenv = np.exp(-4*np.log(2)*(t-t0)**2/FWHM**2)
        return fieldenv

    def envsin2(self,t,function_name,Ncycles,t0,t_cycle):

        fieldenv = np.sin(np.pi * (t) / (Ncycles * t_cycle))**2
        
        return fieldenv

    """ =================================== GENERATE FIELD =================================== """
    def gen_field(self,t):
        """main program constructing the electric field at time t
    
        
            Initial state vectors are eigenfunctions of Hamiltonian `H`. If
                `temp` is None, all eigenfunctions are returned unweighted. If
                `temp` is 0, only lowest energy eigenfunction is returned. If
                `temp` is > 0, all eigenfunctions are returned Boltzmann-
                -weighted.

        Args:
            H : :py:class:`field.CarTens`
                Hamiltonian operator
            temp : float
                Temperature in Kelvin
            thresh : float
                Collective threshold below which to neglect higher energy
                states
            zpe : float
                Zero-point energy, by default lowest eigenvalue of H is taken

        Returns:
            vecs : numpy.ndarray
                Initial state vectors (each row represents an initial state). we return spherical tensor components -1, 0, 1
        """

        
        if self.params['field_form'] == 'analytic':
           
            field_type_function = getattr(self, self.params['field_type']['function_name'] )
            field_env_function = getattr(self, self.params['field_env']['function_name'] )

            if not field_type_function:
                raise NotImplementedError("Method %s not implemented" % self.params['field_type']['field_type'])
            if not field_env_function:
                raise NotImplementedError("Method %s not implemented" % self.params['field_env']['env_type'])


            field_m1 , field_0 , field_p1  = field_type_function(t, **self.params['field_type']) #spherical tensor form
            fieldenv = field_env_function(t, **self.params['field_env'])

            #print(np.shape(output))
            

            #fieldvec2 = np.multiply(fieldenv,fieldz)
            #print("fieldvec2")
            #print(np.shape(fieldvec2))

        elif self.params['field_type'] == 'numerical':
            print("reading field from file")

        Fvec = np.zeros((t.shape[0],3),dtype=complex)


        print(fieldenv.shape)
        print(field_p1.shape)
        #exit()
        Fvec[:,0], Fvec[:,1], Fvec[:,2] = field_m1 * fieldenv , field_0 * fieldenv, field_p1 * fieldenv
        
        print(Fvec.shape)

        Fvec = np.stack(( Fvec[i] for i in range(len(Fvec)) ), axis=1) 


        print(Fvec.shape)
        #exit()
        return Fvec 

