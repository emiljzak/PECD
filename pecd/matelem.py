#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Emil Zak <emil.zak@cfel.de>
#
#import sys
#print(sys.version)#
import numpy as np



class integration():
    """Class containing all integration methods"""

    def __init__(self):
        pass



class potential():
    """Class containing methods for the representation of the PES"""

    def __init__(self):
        pass


class potmat(integration,potential):
    """Class containing methods for the calculation of the potential matrix elements"""

    def __init__(self,Hang):
        self.Hang  = Hang #degree H of angular (ang) quadrature (Lebedev)
    
    def generate(self):
        """calculate full potential matrix"""
        print("hello")

    def calc_potmatelem(self,l1,m1,l2,m2,i1,i2,n1,n2):
        """calculate single element of potential matrix"""
        self.pes


if __name__ == "__main__":      

    potmatrix = potmat(5)    
    potmatrix.generate()

