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


    def gen_field(self,time):
        """main program constructing the electric field"""
        if self.params['field_form'] == 'analytic':
            if self.params['field_name'] == 'LP':
                if self.params['field_env'] == 'gaussian':
                    fieldvec = [0.0 for i in range(3)]
                    fieldvec =  self.params['E0'] * np.array([0.0,0.0,np.cos(self.params['omega']*time)]) #R-CPL

            if self.params['field_name'] == 'LCPL':
                if self.params['field_env'] == 'gaussian':
                    fieldvec = [0.0 for i in range(3)]
                    fieldvec =  self.params['E0'] * np.array([np.cos(self.params['omega']*time) - 1j * np.sin(self.params['omega']*time),-(np.cos(self.params['omega']*time) + 1j * np.sin(self.params['omega']*time)),0.0]) #R-CPL
        
        elif self.params['field_type'] == 'numerical':
            print("reading field from file")

        return np.asarray(fieldvec)