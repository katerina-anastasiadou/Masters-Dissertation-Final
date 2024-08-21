# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:30:47 2024

@author: katan
"""

from helper import get_cutset, get_powerset, find_connected_components
from docplex.mp.model import Model

class MyModel:
    def __init__(self,name_input,data_input):
        
        
        self.model_instance = Model(name_input)

        # Decision variables
        self.x = self.model_instance.binary_var_dict(data_input.E, name = 'x')
        self.y = self.model_instance.binary_var_matrix(data_input.V, data_input.V, name='y')
        

        # Objective function
        self.model_instance.minimize(self.model_instance.sum(self.x[i,j]*data_input.c[i,j] for (i,j) in data_input.E))
        
        # Constraints
        self.model_instance.add_constraints(self.model_instance.sum(self.x[i,j] for (i,j) in get_cutset([j], data_input.E)) == 2 for j in data_input.V)
        
        