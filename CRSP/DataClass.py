# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:37:20 2024

@author: katan
"""

import numpy as np
rnd = np.random
import math

class Data:
    def __init__(self, n_input, width_input, seed_input, V, c_matrix, d_matrix, demand_min=1, demand_max=10, capacity_factor=2):
        
        #self.n = n_input
        self.n = V  # Number of vertices
        self.width = width_input
        self.seed = seed_input
        self.demand_min = demand_min
        self.demand_max = demand_max
        self.capacity_factor = capacity_factor
        self.V = range(V)  # Vertices
        self.c = c_matrix  # Cost matrix c
        self.d = d_matrix  # Cost matrix d
        
    def create_data(self):
        rnd.seed(self.seed)
        self.E = [(i,j) for i in self.V for j in self.V if i>j]
        self.A = [(i,j) for i in self.V for j in self.V]
        self.loc = {i:(rnd.random()*self.width,rnd.random()*self.width) for i in self.V}
        self.demand = {i: np.random.randint(self.demand_min, self.demand_max + 1) for i in self.V}
        self.Qmax = self.capacity_factor * sum(self.demand.values()) / self.n
