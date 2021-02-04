#!/usr/bin/env python
# -*- coding: utf-8; fill-column: 120 -*-
#
import numpy as np

class Field():
    """Class representing electric field"""

    def __init__(self,params):
        self.params = params
        #params['field_form'] = "analytic" #or numerical
        #params['field_name'] = "LP" #RCPL, LCPL, ...
        #params['envelope_name'] = "gaussian" #"static_uniform"


    def gen_field(self,t):
        """main program constructing the electric field"""
        if self.params['field_form'] == 'analytic':

            possibles = globals().copy()
            possibles.update(locals())
            field_type_function = possibles.get(self.params['field_type'])
            field_env_function = possibles.get(self.params['field_env'])
            if not field_type_function:
                raise NotImplementedError("Method %s not implemented" % self.params['field_type'])
                exit()
            if not field_env_function:
                raise NotImplementedError("Method %s not implemented" % self.params['field_env'])
                exit()

            
            fieldvec = self.field_type_function(self.params['field_type'],t)

        elif self.params['field_type'] == 'numerical':
            print("reading field from file")

        return np.asarray(fieldvec)


    def field_CPL(self,field_type, omega, E0, cep=0.0, spherical=True, ftype = "LCPL"):
        fieldvec = [0.0 for i in range(3)]

        if spherical == True:
            if ftype == "LCPL":
                fieldvec = E0 * np.array( [ np.cos( omega * t + cep ) - 1j * np.sin( omega * t + cep ) , - (np.cos( omega * t + cep )  + 1j * np.sin( omega * t + cep ) ), 0.0] ) 
        else:
            exit()


    def field_CPL(self,field_type, omega, E0, cep=0.0, spherical=True, ftype = "LCPL"):
        fieldvec = [0.0 for i in range(3)]

        if spherical == True:
            if ftype == "LCPL":
                fieldvec = E0 * np.array( [ np.cos( omega * t + cep ) - 1j * np.sin( omega * t + cep ) , - (np.cos( omega * t + cep )  + 1j * np.sin( omega * t + cep ) ), 0.0] ) 
        else:
            exit()


            if self.params['field_name'] == 'LCPL':
                if self.params['field_env'] == 'gaussian':
                    fieldenv = np.exp(-4*np.log(2)*(t-t0)**2/width**2)
                    fieldvec = [0.0 for i in range(3)]
                    fieldvec =  self.params['E0'] * 
        