# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:27:01 2024

@author: katan
"""
from cplex.callbacks import LazyConstraintCallback, UserCutCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
from helper import *
import igraph as ig
from itertools import combinations
import time

class Callback_lazy(ConstraintCallbackMixin, LazyConstraintCallback):
    def __init__(self, env):
        """
        Initializes the Callback_lazy class.

        Args:
            model_instance: The CPLEX model instance.
            mdl: An instance of MyModel.
            data: Problem data containing attributes like `E_prime`, `V_prime`, `c_prime`, etc.

        Returns:
            None
        """
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        
        self.visited_vertices = set()  # Initialize an empty set to store visited vertices

        # Initialize the counters for constraints
        self.sec = 0

        self.total_time = 0  # To accumulate total computation time
        
    def __call__(self):
        """
        Callback function to be called for lazy constraint callback.
 
        Returns:
            None
        """        
        
        start_time = self.get_time()  # Start time
        
        print('running lazy callback')
        self.num_calls += 1        
        
        sol_x = self.make_solution_from_vars(self.mdl.x.values())
        sol_y = self.make_solution_from_vars(self.mdl.y.values())
        
        edges_in_solution = [(i, j) for (i, j) in self.problem_data.E if sol_x.get_value(self.mdl.x[i, j]) > 0.9]
        vertices = [i for i in self.problem_data.V if sol_y.get_value(self.mdl.y[i, i]) > 0.9]
 
        self.visited_vertices.update(vertices)  # Store the vertices as visited
        # print("Edges in solution:", edges_in_solution)
        # print("Vertices:", vertices)

        g = ig.Graph()
        g.add_vertices(max(vertices)+1)  # Adding the number of vertices
        g.add_edges(edges_in_solution)
         
        # Get the connected components
        components = g.connected_components(mode='weak')
        
        # Extract the component membership list and adjust back to original IDs
        component_list = [list(comp) for comp in components]
        # print(component_list)
        
        if len(component_list) > 1 :
            for component in component_list:
                # Connectivity constraint
                if len(component) > 2 and 0 not in component:
                    # print("Adding cut for component:", component)
                    ct_cutset = self.mdl.model_instance.sum(self.mdl.x[i, j] for (i, j) in get_cutset(component, self.problem_data.E))
                    for i in component:
                       ct =  ct_cutset >= 2*self.mdl.model_instance.sum(self.mdl.y[i, j] for j in component)
                       ct_cpx = self.linear_ct_to_cplex(ct)
                       self.add(ct_cpx[0],ct_cpx[1],ct_cpx[2])
                       
                       self.sec += 1  # Increment sec counter
                       
                       
        # End of callback logic
        end_time = self.get_time()  # End time
        elapsed_time = end_time - start_time
        self.total_time += elapsed_time  # Accumulate time

        # print(f'Lazy callback execution time: {elapsed_time:.4f} seconds')
        # print(f'Total time in lazy callback so far: {self.total_time:.4f} seconds')
                       
                       
class Callback_user(ConstraintCallbackMixin, UserCutCallback):
    def __init__(self, env):
        """
        Initializes the Callback_user class.

        Args:
            model_instance: The CPLEX model instance.
            mdl: An instance of MyModel.
            data: Problem data containing attributes like `E_prime`, `V_prime`, `c_prime`, etc.

        Returns:
            None
        """
        UserCutCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        
        self.visited_vertices = set()  # Initialize an empty set to store visited vertices
        
        self.mip_gap = 0
        self.lb = 0
        self.ub = 0

        # Initialize counters
        self.sec = 0
        self.mat2 = 0
        self.cover = 0
        
        self.total_nodes_examined = 0  # Variable to track the total number of nodes examined

        self.total_time = 0  # To accumulate total computation time
        
    def __call__(self):
        """
        Callback function to be called for user cut callback.

        Returns:
            None
        """
        
        start_time = self.get_time()  # Start time
        
        # Update the total number of nodes examined
        self.total_nodes_examined = self.get_num_nodes()
        
        if self.get_current_node_depth() == 0:
            self.mip_gap_root_node = self.get_MIP_relative_gap()
            print(f' mip gap {self.mip_gap_root_node}')
            self.mip_gap = self.mip_gap_root_node
            print(f' upper bound {self.get_incumbent_objective_value()}')
            self.ub = self.get_incumbent_objective_value()
            print(f' lower bound {self.get_best_objective_value()}') 
            self.lb = self.get_best_objective_value()
        
        #print('running user callback')
        self.num_calls += 1

        sol_x = self.make_solution_from_vars(self.mdl.x.values())
        sol_y = self.make_solution_from_vars(self.mdl.y.values())
        
        edges_in_solution = [(i, j) for (i, j) in self.problem_data.E if sol_x.get_value(self.mdl.x[i, j]) > 0.0000001]
        vertices = [i for i in self.problem_data.V if sol_y.get_value(self.mdl.y[i, i]) > 0.000001]
        
        self.visited_vertices.update(vertices)  # Store the vertices as visited
        
        edges_map = map_edges(vertices, edges_in_solution)
        
        # print("Edges in solution:", edges_in_solution)
        # print("mapped Edges in solution:", edges_map)
        # print("Vertices:", vertices)
        
        # Connectivity constraints
        g = ig.Graph()
        g.add_vertices(vertices)
        g.add_edges(edges_map)
         
        # Get the connected components
        components = g.connected_components(mode='weak')
        component_list = [list(comp) for comp in components]
        # print(component_list)

        if len(component_list) > 1:
            for component in component_list:
                component_unmapped = [vertices[i] for i in component]
                if len(component_unmapped) > 2 and 0 not in component_unmapped:
                    #print(component_unmapped)
                    for i in component_unmapped:
                        ct_cutset = self.mdl.model_instance.sum(self.mdl.x[i, j] for (i, j) in get_cutset(component_unmapped, self.problem_data.E))
                        ct = ct_cutset >= 2 * self.mdl.model_instance.sum(self.mdl.y[i, j] for j in component_unmapped)
                        ct_cpx = self.linear_ct_to_cplex(ct)
                        self.add(ct_cpx[0], ct_cpx[1], ct_cpx[2])                    
        else:
            # print('min cut here')
            capacity = [max(1, sol_x.get_value(self.mdl.x[i, j])) for (i, j) in edges_in_solution]
            g.es["capacity"] = capacity
            cut = g.mincut()
            # print('run min cut')
            g.add_vertex(self.problem_data.n)
            for v in vertices:
                if v != 0:
                    dummy_edges = []
                    dummy_capacity = []                    
                    for j in self.problem_data.V:
                        if sol_y.get_value(self.mdl.y[v, j]) > 0.000001:                            
                            dummy_edges.append((j,self.problem_data.n))
                            dummy_capacity.append(sol_y.get_value(self.mdl.y[v, j]))
                    # print('finished loop')
                    dummy_vertices = vertices + [self.problem_data.n]
                    edges_map_dummy = map_edges(dummy_vertices, dummy_edges)
                    #print(dummy_edges)
                    #print(edges_map_dummy)
                    #print([v for v in g.vs])
                    g.add_edges(edges_map_dummy)
                    #print(len(dummy_vertices))
                    cut = g.mincut(0, len(vertices))
                    value = cut.value
                    partition = cut.partition
                    # print("Value:", value)
                    # print("Partition:",  partition)
                    g.delete_edges(edges_map_dummy)
                    if value < 2 * sol_y.get_value(self.mdl.y[v, v]):
                         # print('violated sec add cut')
                         for component in partition:
                             # print("Component:", component)
                             component_unmapped = [dummy_vertices[i] for i in component]
                             # print("Component unmapped:", component_unmapped)
                             if self.problem_data.n in component_unmapped:
                                component_unmapped.remove(self.problem_data.n)
                             if len(component_unmapped) > 2 and 0 not in component_unmapped:
                                 ct_cutset = self.mdl.model_instance.sum(self.mdl.x[i, j] for (i, j) in get_cutset(component_unmapped, self.problem_data.E))
                                 ct = ct_cutset >= 2 * self.mdl.model_instance.sum(self.mdl.y[v, j] for j in component_unmapped)
                                 #print(self.mdl.y[v, j] for j in component_unmapped)
                                 ct_cpx = self.linear_ct_to_cplex(ct)
                                 self.add(ct_cpx[0], ct_cpx[1], ct_cpx[2])
                                 # solve a global min cut problem and if the value of the cut is less than 2 we have a violated cut
                                 
                                 self.sec += 1  # Increment sec counter
                                 
        # 2-matching inequalities      
        int_edges = [(i, j) for (i, j) in self.problem_data.E if sol_x.get_value(self.mdl.x[i, j]) >= 0.999999]
                
        G_star = ig.Graph()
        
        E_star = [(i, j) for (i, j) in self.problem_data.E if sol_x.get_value(self.mdl.x[i, j]) > 0.0000001 and sol_x.get_value(self.mdl.x[i, j]) < 0.999999]
        
        V = set()
        for (i,j) in E_star:
            V.add(i)
            V.add(j)
        V = list(V)
        #print(f' V {V}')
        #print(f' E_star {E_star}')
        edges_map = map_edges(V, E_star)
        #print(f' mapped edges {edges_map}')
        G_star.add_vertices(V)
        G_star.add_edges(edges_map)

        # Find connected components in G*
        components_star = G_star.connected_components(mode='weak')
        component_list_star = [list(comp) for comp in components_star]
        #print("Components in G*:", component_list_star)
        
        # Iterate over each component H in G*
        for component in component_list_star:
            component_unmapped = [V[i] for i in component]
            T = []
            for (i, j) in int_edges:
                if i in component_unmapped or j in component_unmapped:
                    T.append((i, j))
            # print(T)
            # Process each component
            if len(component_unmapped) > 2 and len(T) % 2 == 1:
                sum_x_EH = self.mdl.model_instance.sum(self.mdl.x[i, j] for (i, j) in self.problem_data.E if i in component_unmapped and j in component_unmapped)
                sum_x_T = self.mdl.model_instance.sum(self.mdl.x[i, j] for (i, j) in T)
                sum_y_H = self.mdl.model_instance.sum(self.mdl.y[i, i] for i in component_unmapped)
                half_T_len = (len(T) - 1) / 2
                inequality = sum_x_EH + sum_x_T <= sum_y_H + half_T_len
                
                # Blossom inequalities
                # x_EH_T = self.mdl.model_instance.sum(self.mdl.x[i, j] for (i, j) in self.problem_data.E if i in component_unmapped and j in component_unmapped and (i, j) not in T)
                # x_T = self.mdl.model_instance.sum(self.mdl.x[i, j] for (i, j) in T)
                # inequality = x_EH_T + (len(T) - x_T) >= 1
                
                ct_cpx = self.linear_ct_to_cplex(inequality)
                self.add(ct_cpx[0], ct_cpx[1], ct_cpx[2])
                #print(f"Added 2-matching constraint: {inequality}")
                
                self.mat2 += 1  # Increment 2mat counter when this constraint is added
                
                # Remove used integer edges from int_edges
                for edge in T:
                    if edge in int_edges:
                        int_edges.remove(edge)
                        
        
        # Separation of cover inequalities
        
        
        # Iterate over each vertex j in V
        # for j in self.problem_data.V:
        #     # Step 1: Identify neighbors N where y*ij > 0
        #     neighbors = [i for i in self.problem_data.V if sol_y.get_value(self.mdl.y[i, j]) > 0]
            
        #     # Step 2: Initialize variables for the heuristic
        #     z = {i: 0 for i in neighbors}  # Initialize z_i variables to 0
        #     max_obj = 0  # Initialize objective value
        #     total_demand = 0  # Initialize total demand
            
        #     # Step 3: Sort neighbors by the heuristic criteria (e.g., non-increasing (1 - y*ij))
        #     sorted_neighbors = sorted(neighbors, key=lambda i: (1 - sol_y.get_value(self.mdl.y[i, j])) / self.problem_data.demand[i], reverse=True)
            
        #     # Step 4: Greedily select items for the cover
        #     for i in sorted_neighbors:
        #         # Check the constraint: fi * z_i >= Qmax * yjj + epsilon
        #         if self.problem_data.demand[i] >= self.problem_data.Qmax * sol_y.get_value(self.mdl.y[j, j]) + 1e-6:
        #             z[i] = 1
        #             total_demand += self.problem_data.demand[i]
        #             max_obj += (1 - sol_y.get_value(self.mdl.y[i, j]))
            
        #     # Step 5: Check if the objective value is less than 1
        #     if max_obj < 1:
        #         # Define the violated cover set C = {i: z_i = 1}
        #         C = [i for i in neighbors if z[i] == 1]
        #         #print(f'cover: {C}')
                
        #         # Step 6: Add the corresponding constraint if C is non-empty
        #         if C:
        #             violated_cut = self.mdl.model_instance.sum(self.mdl.y[i, j] for i in C) <= len(C) - 1
                    
        #             # Safety check: Ensure the constraint is not too restrictive
        #             #if len(C) > 1:
        #             ct_cpx = self.linear_ct_to_cplex(violated_cut)
        #             self.add(ct_cpx[0], ct_cpx[1], ct_cpx[2])
        #             #print(f"Added cover constraint: {violated_cut}")
        #             self.cover += 1


        # Iterate over each vertex j in V
        for j in self.problem_data.V:
            # Step 1: Identify neighbors N where y*ij > 0
            neighbors = [i for i in self.problem_data.V if sol_y.get_value(self.mdl.y[i, j]) > 0]
            
            # Step 2: Initialize variables for the heuristic
            z = {i: 0 for i in neighbors}  # Initialize z_i variables to 0
            max_obj = 0  # Initialize objective value
            total_demand = 0  # Initialize total demand
            
            # Step 3: Sort neighbors by the heuristic criteria (e.g., non-increasing (1 - y*ij))
            sorted_neighbors = sorted(neighbors, key=lambda i: (1 - sol_y.get_value(self.mdl.y[i, j])) / self.problem_data.demand[i], reverse=True)
            
            # Step 4: Greedily select items for the cover
            for i in sorted_neighbors:
                # Calculate the incremental demand if we include item i in the cover
                current_demand = total_demand + self.problem_data.demand[i]
                
                # Check the constraint: total_demand >= Qmax * yjj + epsilon
                if current_demand >= self.problem_data.Qmax * sol_y.get_value(self.mdl.y[j, j]) + 1e-6:
                    z[i] = 1
                    total_demand = current_demand  # Update the total demand
                    max_obj += (1 - sol_y.get_value(self.mdl.y[i, j]))  # This is the max of (1 - y_ij) * z_i
                    
                    # Stop adding more items since the constraint is satisfied
                    break
            
            # Step 5: Check if the objective value is less than 1
            if max_obj < 1:
                # Define the violated cover set C = {i: z_i = 1}
                C = [i for i in neighbors if z[i] == 1]
                #print(f'cover: {C}')
                
                # Step 6: Add the corresponding constraint if C is non-empty
                if C:
                    violated_cut = self.mdl.model_instance.sum(self.mdl.y[i, j] for i in C) <= len(C) - 1
                    ct_cpx = self.linear_ct_to_cplex(violated_cut)
                    self.add(ct_cpx[0], ct_cpx[1], ct_cpx[2])
                    self.cover += 1

        


          
        # # Iterate over each vertex j in V
        # for j in self.problem_data.V:
        #     # Step 1: Identify neighbors N where y*ij > 0
        #     neighbors = [i for i in self.problem_data.V if sol_y.get_value(self.mdl.y[i, j]) > 0]
            
        #     # Step 2: Solve the 0-1 Knapsack problem to maximize (1 - y*ij)zi
        #     z = {i: 0 for i in neighbors}  # Initialize z_i variables to 0
        #     max_obj = 0 # Initialize obj
            
        #     # Heuristic: Sort neighbors by (1 - y*ij) in descending order
        #     for i in sorted(neighbors, key=lambda i: (1 - sol_y.get_value(self.mdl.y[i, j])) / self.problem_data.demand[i], reverse=True):
        #         # Check the constraint: fi * zi >= Qmax * yjj + epsilon
        #         if self.problem_data.demand[i] >= self.problem_data.Qmax * sol_y.get_value(self.mdl.y[j, j]) + 1e-6:
        #             z[i] = 1
        #             max_obj += (1 - sol_y.get_value(self.mdl.y[i, j])) * z[i]
        #             break
                
        #     # Step 3: Check if the objective value is less than 1
        #     if max_obj < 1:
        
        #         # Define the violated cut C = {i: z_i = 1}
        #         C = [i for i in neighbors if z[i] == 1]
        #         #print(f'cover: {C}')
                
        #         # Ensure C is not empty before adding the constraint
        #         if C:# and len(C) > 1:  # Ensure at least one element in C to make the cut meaningful
        #             violated_cut = self.mdl.model_instance.sum(self.mdl.y[i, j] for i in C) <= len(C) - 1
        #             ct_cpx = self.linear_ct_to_cplex(violated_cut)
        #             self.add(ct_cpx[0], ct_cpx[1], ct_cpx[2])
        #             #print(f"Added cover constraint: {violated_cut}")
        #             self.cover += 1
        
        
        

        # for j in self.problem_data.V:
        #     # Step 1: Identify neighbors N where y*ij > 0
        #     neighbors = [i for i in self.problem_data.V if sol_y.get_value(self.mdl.y[i, j]) > 0]
            
        #     # Step 2: Initialize variables for the heuristic
        #     z = {i: 0 for i in neighbors}  # Initialize z_i variables to 0            
        #     max_obj = -float('inf')  # Initialize the objective value to a very low number
        #     best_i = None  # Variable to store the best candidate i
            
        #     # Step 3: Sort neighbors by the heuristic criteria (e.g., non-increasing (1 - y*ij))
        #     sorted_neighbors = sorted(neighbors, key=lambda i: (1 - sol_y.get_value(self.mdl.y[i, j])) / self.problem_data.demand[i], reverse=True)
                       
        #     # Iterate over neighbors to find the best candidate
        #     for i in sorted_neighbors:
        #         # Calculate the potential objective for this i
        #         current_obj = (1 - sol_y.get_value(self.mdl.y[i, j])) * 1  # Since z_i will be 1 if chosen
                
        #         # Check the constraint: fi * 1 >= Qmax * yjj + epsilon
        #         if self.problem_data.demand[i] >= self.problem_data.Qmax * sol_y.get_value(self.mdl.y[j, j]) + 1e-6:
        #             # If this objective is better than the previous best, update it
        #             if current_obj > max_obj:
        #                 max_obj = current_obj
        #                 best_i = i
        
        #     # Step 3: Check if the objective value is less than 1
        #     if max_obj < 1 and best_i is not None:
        #         # Define the violated cover set C = {best_i}
        #         C = [best_i]
        #         #print(f'cover: {C}')
        
        #         if C:# and len(C) > 1:
        #             # Add the corresponding constraint
        #             violated_cut = self.mdl.model_instance.sum(self.mdl.y[i, j] for i in C) <= len(C) - 1
                    
        #             # Safety check: Ensure the constraint is not too restrictive
        #             ct_cpx = self.linear_ct_to_cplex(violated_cut)
        #             self.add(ct_cpx[0], ct_cpx[1], ct_cpx[2])
        #             #print(f"Added cover constraint: {violated_cut}")
        #             self.cover += 1
            
            
        # for j in self.problem_data.V:
        #     neighbors = [i for i in self.problem_data.V if sol_y.get_value(self.mdl.y[i, j]) > 0]
        #     z = {i: 0 for i in neighbors}
        #     total_demand = 0
        #     max_obj = 0
        
        #     for i in sorted(neighbors, key=lambda i: (1 - sol_y.get_value(self.mdl.y[i, j])), reverse=True):
        #         # After setting z[i] = 1, check the feasibility constraint
        #         z[i] = 1
        #         if total_demand + self.problem_data.demand[i] >= self.problem_data.Qmax * sol_y.get_value(self.mdl.y[j, j]) + 1e-6:
        #             break
        #         total_demand += self.problem_data.demand[i]
        #         max_obj += (1 - sol_y.get_value(self.mdl.y[i, j])) * z[i]
                
        #     if max_obj < 1:# and total_demand < self.problem_data.Qmax * sol_y.get_value(self.mdl.y[j, j]) + 1e-6:
            
        #         C = [i for i in neighbors if z[i] == 1]
        #         print(f'cover: {C}')
        #         if C:# and len(C) > 1:
        #             violated_cut = self.mdl.model_instance.sum(self.mdl.y[i, j] for i in C) <= len(C) - 1
        #             ct_cpx = self.linear_ct_to_cplex(violated_cut)
        #             self.add(ct_cpx[0], ct_cpx[1], ct_cpx[2])
        #             print(f"Added cover constraint: {violated_cut}")
        #             self.cover += 1
        
        
        # End of callback logic
        end_time = self.get_time()  # End time
        elapsed_time = end_time - start_time
        self.total_time += elapsed_time  # Accumulate time

        # print(f'User callback execution time: {elapsed_time:.4f} seconds')
        # print(f'Total time in user callback so far: {self.total_time:.4f} seconds')




