# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:28:08 2024

@author: katan
"""

import numpy as np
rnd = np.random
import math

class Data:
    def __init__(self, n_input, width_input, seed_input):
        
        self.n = n_input
        self.width = width_input
        self.seed = seed_input
        
    def create_data(self):
        rnd.seed(self.seed)
        self.V = range(self.n)
        self.E = [(i,j) for i in self.V for j in self.V if i>j]
        self.loc = {i:(rnd.random()*self.width,rnd.random()*self.width) for i in self.V}
        self.c = {(i,j): math.hypot(self.loc[i][0]-self.loc[j][0],self.loc[i][1]-self.loc[j][1]) for (i,j) in self.E}