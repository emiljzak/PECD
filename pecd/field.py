#!/usr/bin/env python
# -*- coding: utf-8; fill-column: 120 -*-
#

class Field(object):
    """Class representing electric field"""

    def __init__(self,params):
        self.params = params
    def gen_field(self):
        """main program constructing the electric field"""
        if self.params['field_type'] == 'analytic':
            if self.params['field'] == 'gaussian':
                fieldvec = [0.0 for i in range(3)]
        if self.params['field'] == 'static_uniform':
                fieldvec = [0.0,0.0,self.params['E0']]


        return fieldvec