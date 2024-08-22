# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:46:31 2024

@author: katan
"""
from helper import get_cutset, find_connected_components, get_edges
from docplex.mp.model import Model

class MyModel:
    def __init__(self,name_input,data_input):
        
        
        self.model_instance = Model(name_input)

        # Decision variables (7), (Integrality for yij)
        self.x = self.model_instance.binary_var_dict(data_input.E, name = 'x')
        self.y = self.model_instance.binary_var_matrix(data_input.V, data_input.V, name='y')
        

        # Objective function (3)
        # self.model_instance.minimize(
        #     self.model_instance.sum(self.x[i, j] * data_input.c[i, j] for (i,j) in data_input.E) +
        #     self.model_instance.sum(self.y[i, j] * data_input.a[i, j] for (i,j) in data_input.A))
        
        self.model_instance.minimize(
            self.model_instance.sum(self.x[i, j] * data_input.c[i, j] for (i,j) in data_input.E) +
            self.model_instance.sum(self.y[i, j] * data_input.d[i, j] for (i,j) in data_input.A))

        # Constraints
        # Degree constraint (4): Ensure the sum of edges adjacent to i is at least 2 times the sum of y[i,i] for all i
        self.model_instance.add_constraints(
            self.model_instance.sum(self.x[i,j] for (i,j) in get_cutset([i], data_input.E)) == 2 * self.y[i,i] 
            for i in data_input.V)
        
        # Assignment constraint (5): Each i must be assigned to exactly one j, except for i=0
        self.model_instance.add_constraints(
            self.model_instance.sum(self.y[i, j] for j in data_input.V) == 1
            for i in data_input.V if i != 0)
        
        # Ensure y[0, 0] is 1 (9)
        self.model_instance.add_constraint(self.y[0,0] == 1)
        
        # Ensure y[0, j] is 0 for all j except j=0 (10)
        self.model_instance.add_constraints(
            self.y[0, j]  == 0
            for j in data_input.V if j !=0 )
        
        # Ensure y[i, j] is less than or equal to y[j, j] for all i, j
        # self.model_instance.add_constraints(self.y[i,j] <= self.y[j,j] for i in data_input.V for j in data_input.V)
        
        # Additional constraint (12): x[i, j] + y[i, j] <= y[j, j] for all pairs of vertices i, j
        # where S = {v_i, v_j} and S \ {v_1} is true
        self.model_instance.add_constraints(
            self.x[e] + self.y[i, j] <= self.y[j, j] 
            for i in data_input.V for j in data_input.V for e in get_edges([i,j],data_input.E)
            if i != 0 and j != 0 and i != j 
        )
        

        # Constraint (13): x[0, i] <= y[i, i] for all vi in V \ {v1}
        self.model_instance.add_constraints(
            self.x[e] <= self.y[i, i] for i in data_input.V for e in get_edges([0,i],data_input.E) if i != 0
        )
        
        
        # Capacity constraint
        # Cover inequality: Total demand assigned to j should not exceed Qmax * y[j, j]
        self.model_instance.add_constraints(
            self.model_instance.sum(data_input.demand[i] * self.y[i, j] for i in data_input.V) <= data_input.Qmax * self.y[j, j]
            for j in data_input.V
        )
        
        
        
        
        
        