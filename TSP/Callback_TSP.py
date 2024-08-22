# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:01:19 2024

@author: katan
"""

from cplex.callbacks import LazyConstraintCallback, UserCutCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
from helper import *
import igraph as ig

class Callback_lazy(ConstraintCallbackMixin, LazyConstraintCallback):
    def __init__(self, env):
        """
        Initializes the Callback_lazy class.
 
        Args:
            env: CPLEX environment.
 
        Returns:
            None
        """
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
    def __call__(self):
        """
        Callback function to be called for lazy constraint callback.
 
        Returns:
            None
        """
        print('running lazy callback')
        self.num_calls += 1
        sol_x = self.make_solution_from_vars(self.mdl.x.values())
    
        edges_in_solution = [(i, j) for (i, j) in self.problem_data.E if sol_x.get_value(self.mdl.x[i,j]) > 0.9]
        
        g = ig.Graph()
        g.add_vertices(len(self.problem_data.V))  # Adding the number of vertices
        g.add_edges(edges_in_solution)
         
        # Get the connected components
        components = g.connected_components(mode='weak')
        
        # Extract the component membership list and adjust back to original IDs
        component_list = [[v for v in comp] for comp in components]
      
        if len(component_list) > 1:
            for component in component_list:
                # Connectivity constraint
                if len(component)>2 and 0 not in component:
                   print(component)
                   ct_cutset = self.mdl.model_instance.sum(self.mdl.x[i, j] for (i, j) in get_cutset(component, self.problem_data.E))
                   
                   ct =  ct_cutset >= 2
                   ct_cpx = self.linear_ct_to_cplex(ct)
                   self.add(ct_cpx[0],ct_cpx[1],ct_cpx[2])
                   
class Callback_user(ConstraintCallbackMixin, UserCutCallback):
    def __init__(self, env):
        """
        Initializes the Callback_user class.

        Args:
            env: CPLEX environment.

        Returns:
            None
        """
        UserCutCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        
    def __call__(self):
        """
        Callback function to be called for user cut callback.

        Returns:
            None
        """
        print('running user callback')
        self.num_calls += 1
        sol_x = self.make_solution_from_vars(self.mdl.x.values())
        
        edges_in_solution = [(i, j) for (i, j) in self.problem_data.E if sol_x.get_value(self.mdl.x[i,j]) > 0.0000001]
        
        g = ig.Graph()
        g.add_vertices(len(self.problem_data.V))  # Adding the number of vertices
        g.add_edges(edges_in_solution)
         
        # Get the connected components
        components = g.connected_components(mode='weak')
        
        # Extract the component membership list and adjust back to original IDs
        component_list = [[v for v in comp] for comp in components]
      
        if len(component_list) > 1:
            for component in component_list:
                # Connectivity constraint
                if len(component)>2 and 0 not in component:
                   print(component)
                   ct_cutset = self.mdl.model_instance.sum(self.mdl.x[i, j] for (i, j) in get_cutset(component, self.problem_data.E))
                   
                   ct =  ct_cutset >= 2
                   ct_cpx = self.linear_ct_to_cplex(ct)
                   self.add(ct_cpx[0],ct_cpx[1],ct_cpx[2])
        else:
            print('min cut here')
            capacity = [max(1,sol_x.get_value(self.mdl.x[i,j])) for (i,j) in edges_in_solution]
            g.es["capacity"] = capacity
            cut = g.mincut()
            print('run min cut')
            value = cut.value
            partition = cut.partition
            print(value)
            print(partition)
            if value < 2:
                print('violated sec add cut')
                for component in partition:
                    # Connectivity constraint
                    if len(component)>2 and 0 not in component:
                       print(component)
                       ct_cutset = self.mdl.model_instance.sum(self.mdl.x[i, j] for (i, j) in get_cutset(component, self.problem_data.E))
                       
                       ct =  ct_cutset >= 2
                       ct_cpx = self.linear_ct_to_cplex(ct)
                       self.add(ct_cpx[0],ct_cpx[1],ct_cpx[2])
