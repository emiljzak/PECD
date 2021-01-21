#!/usr/bin/env python
# -*- coding: utf-8; fill-column: 120 -*-
#
import numpy as np

class Field():
    """Class representing electric field"""

    def __init__(self,params):
        self.params = params

    def gen_field(self):
        """main program constructing the electric field"""
        if self.params['field_type'] == 'analytic':
            if self.params['field_env'] == 'gaussian':
                fieldvec = [0.0 for i in range(3)]
            elif self.params['field_env'] == 'static_uniform':
                fieldvec = [0.0,0.0,self.params['E0']]
        elif self.params['field_type'] == 'numerical':
            print("reading field from file")

        return np.asarray(fieldvec)