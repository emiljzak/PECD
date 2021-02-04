#!/usr/bin/env python
# -*- coding: utf-8; fill-column: 120 -*-
#
import numpy as np

class Field():
    """Class representing electric field"""

    def __init__(self,params):
        self.params = params

    """ =================================== FIELD TYPES =================================== """
    def fieldCPL(self,t, omega, E0, CEP0, spherical, typef):
   
        fieldvec = [0.0 for i in range(3)]

        if spherical == True:
            if typef == "LCPL":

                fieldvec = E0 * np.array( [ np.cos( omega * t + CEP0 ) - 1j * np.sin( omega * t + CEP0 ) , - (np.cos( omega * t + CEP0 )  + 1j * np.sin( omega * t + CEP0 ) ), 0.0] ) 
        else:

        return fieldvec

    def printdummy(self,a):
        print("this is a: "+str(a))


    """ =================================== FIELD ENVELOPES =================================== """
    def envgaussian(self,t,fwhm,t0):
        fieldenv = 0.0
        fieldenv = np.exp(-4*np.log(2)*(t-t0)**2/fwhm**2)
        return fieldenv


    """ =================================== GENERATE FIELD =================================== """
    def gen_field(self,t):
        """main program constructing the electric field at time t"""
        if self.params['field_form'] == 'analytic':
           
            field_type_function = getattr(self, self.params['field_type']['function_name'] )
            field_env_function = getattr(self, self.params['field_env']['function_name'] )

            if not field_type_function:
                raise NotImplementedError("Method %s not implemented" % self.params['field_type']['field_type'])
            if not field_env_function:
                raise NotImplementedError("Method %s not implemented" % self.params['field_env']['env_type'])

            self.params['field_type'].pop('function_name')

            fieldvec = field_type_function(t,**self.params['field_type'])
            exit()
            fieldenv = field_env_function(t, **self.params['field_env'])
            print(fieldvec,fieldenv)

        elif self.params['field_type'] == 'numerical':
            print("reading field from file")

        return np.asarray(fieldvec)

