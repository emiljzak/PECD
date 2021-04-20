#!/usr/bin/env python
# -*- coding: utf-8; fill-column: 120 -*-
#
import numpy as np

class Field():
    """Class representing electric field"""

    def __init__(self,params):
        self.params = params

    """ =================================== FIELD TYPES =================================== """
    def fieldCPL(self,t, function_name, omega, E0, CEP0, spherical, typef):
        #all vectors are returned in spherical tensor form -1, 0, 1 order
        if spherical == True:
            if typef == "LCPL":

                fieldvec = E0 * np.array( [ np.cos( omega * t + CEP0 ) - 1j * np.sin( omega * t + CEP0 ) , 0.0, - (np.cos( omega * t + CEP0 )  + 1j * np.sin( omega * t + CEP0 ) )] ) 
        else:
            if typef == "LCPL":
                fieldvec = E0 * np.array( [ np.cos( omega * t + CEP0 ) - 1j * np.sin( omega * t + CEP0 ) , 0.0, - (np.cos( omega * t + CEP0 )  + 1j * np.sin( omega * t + CEP0 ) )] ) 
        return fieldvec

    def fieldLP(self, t, function_name, omega, E0, CEP0):
        #fieldvec = t
        #print("shapes in fieldLP")
        #print(np.shape(t))
        #print(np.shape(np.cos( omega * t + CEP0 )))
        return 0.0, np.cos( omega * t + CEP0 ), 0.0


    """ =================================== FIELD ENVELOPES =================================== """
    def envgaussian(self,t,function_name,FWHM,t0):
        fieldenv = np.exp(-4*np.log(2)*(t-t0)**2/FWHM**2)
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


            field_m1 , field_0 , field_p1  = field_type_function(t, **self.params['field_type']) #spherical tensor form
            fieldenv = field_env_function(t, **self.params['field_env'])

            #print(np.shape(output))
            

            #fieldvec2 = np.multiply(fieldenv,fieldz)
            #print("fieldvec2")
            #print(np.shape(fieldvec2))

        elif self.params['field_type'] == 'numerical':
            print("reading field from file")

        return field_m1 * fieldenv , field_0 * fieldenv, field_p1 * fieldenv #note that we return spherical tensor components -1, 0, 1

