# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:32:52 2022

@author: 20771401
"""

from DataClass import Data
from Model import MyModel
from helper import *
from Callback import Callback_lazy, Callback_user
import igraph as ig
import pandas as pd
import time

#%% Input data

def calculate_p_star(visited_vertices, total_vertices):
    return (len(visited_vertices) / total_vertices) * 100

def calculate_percent_lb(solution, cb_user):
    optimal_value = solution.get_objective_value()
    lb = cb_user.lb
        
    if lb is None:
        raise ValueError("Lower bound (LB) is not set.")
    
    if optimal_value == 0:
        raise ValueError("Optimal value is zero, cannot compute %_LB.")
    
    percent_lb = (lb / optimal_value) * 100
    return percent_lb

def calculate_percent_ub(solution, cb_user):
    optimal_value = solution.get_objective_value()
    ub = cb_user.ub

    if ub is None:
        raise ValueError("Upper bound (UB) is not set.")
    
    if optimal_value == 0:
        raise ValueError("Optimal value is zero, cannot compute %_UB.")
    
    percent_ub = (ub / optimal_value) * 100
    
    # Calculate the percentage ratio %-UB
    # percent_ub = ((ub - optimal_value) / optimal_value) * 100
    # print(percent_ub)
    return percent_ub

def format_time(total_seconds):
    """"
    Convert a float value representing seconds into hh:mm:ss format.
    
    :param total_seconds: Time in seconds as a float.
    :return: Time in hh:mm:ss format as a string.
    """
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'

n = 40
dataset = 1
width = 100

V_values = [125]#[50, 75, 100, 125, 150, 175, 200]
alpha_values = [3, 5, 7, 9]
num_instances = 10

results = []
best_sol = []

# Add a new constant for the maximum allowed time in seconds
MAX_TIME = 3600  # 60 minutes

for V in V_values:    
    for alpha in alpha_values:
        summary_stats = {
            'V': V,
            'alpha': alpha,
            'succ': 0,
            'p_star': 0,
            'opt': 0,
            '%_LB': 0,
            '%_UB': 0,
            'sec': 0,
            '2mat': 0,
            'cover': 0,
            'mip gap': 0,
            'nodes': 0,
            'time': 0
        }
        
        # Initialize best solution tracking
        best_instance_index = None
        best_objective_value = float('inf')  # Assume a large number
        best_percent_ub = None  # Track the best UB percent
        
        for i in range(num_instances):
            try:
                start_time = time.time()  # Start the timer
                
                vertices, l_matrix, c_matrix, d_matrix = generate_instance(V, alpha)
                print(f"Vertices: {vertices}")
                print(f"Cost matrix (c_matrix):\n{c_matrix}")
                print(f"Distance matrix (d_matrix):\n{d_matrix}")
    
                p  = Data(n, width, dataset, V, c_matrix, d_matrix)
                p.create_data()
                
                mdl = MyModel("RSP",p)
                
                cb_lazy = mdl.model_instance.register_callback(Callback_lazy)
                cb_lazy.mdl = mdl
                cb_lazy.problem_data = p
                cb_lazy.num_calls = 0
                
                cb_user = mdl.model_instance.register_callback(Callback_user)
                cb_user.mdl = mdl
                cb_user.problem_data = p
                cb_user.num_calls = 0
                
                # Solve the model
                # solution = mdl.model_instance.solve(log_output=True)
                
                # Solve the model
                solution = None
                elapsed_time = 0
                
                while elapsed_time < MAX_TIME:
                    solution = mdl.model_instance.solve(log_output=True)
                    elapsed_time = time.time() - start_time
                    if solution:  # Break the loop if a solution is found
                        break
                
                if elapsed_time >= MAX_TIME:
                    print(f"Instance {i} with V={V} and alpha={alpha} terminated after exceeding 60 minutes.")
                    continue  # Skip the rest of the loop for this instance
                
                # Check if a solution is found
                if solution:
                    # Print the objective value
                    #print("Objective value:", solution.get_objective_value())
                    
                    current_objective_value = solution.get_objective_value()
                    print("Objective value:", current_objective_value)
                    
                    # Calculate %_UB for the current solution
                    try:
                        current_percent_ub = calculate_percent_ub(solution, cb_user)
                        print(current_percent_ub)
                    except ValueError as e:
                        print(f"Warning: {e}")
                        current_percent_ub = None
                    
                    # Update best instance if current one is better
                    if current_objective_value < best_objective_value:
                        best_objective_value = current_objective_value
                        best_instance_index = i
                        best_percent_ub = current_percent_ub
                else:
                    print("No solution found.")
                    
                # Initialize the statistics dictionary
                stats = {
                    'V': V,
                    'alpha': alpha,
                    'succ': 1 if solution else 0,
                    'p_star': None,
                    'opt': None,
                    '%_LB': None,
                    '%_UB': None,
                    'sec': None,
                    '2mat': None,
                    'cover': None,
                    'mip gap': None,
                    'nodes': None,
                    'time': None
                }
    
                if solution:
                    visited_vertices = cb_lazy.visited_vertices.union(cb_user.visited_vertices)
                    stats['p_star'] = calculate_p_star(visited_vertices, V)
                    stats['opt'] = solution.get_objective_value()
                    
                    try:
                        stats['%_LB'] = calculate_percent_lb(solution, cb_user)
                    except ValueError as e:
                        print(f"Warning: {e}")
                        
                    try:
                        stats['%_UB'] = calculate_percent_ub(solution, cb_user)
                    except ValueError as e:
                        print(f"Warning: {e}")
                    
                    stats['sec'] = cb_lazy.sec + cb_user.sec
                    stats['2mat'] = cb_user.mat2
                    stats['cover'] = cb_user.cover
                    stats['mip gap'] = cb_user.mip_gap
                    stats['nodes'] = cb_user.total_nodes_examined
                    stats['time'] = cb_lazy.total_time + cb_user.total_time
                
                # Plot the solution
                plot_sol(p,mdl,V,alpha,i)
                
                # Accumulate the results
                summary_stats['succ'] += stats.get('succ', 0)
                summary_stats['p_star'] += stats.get('p_star', 0)
                summary_stats['opt'] += stats.get('opt', 0)
                summary_stats['%_LB'] += stats.get('%_LB', 0)
                summary_stats['%_UB'] += stats.get('%_UB', 0)
                summary_stats['sec'] += stats.get('sec', 0)
                summary_stats['2mat'] += stats.get('2mat', 0)
                summary_stats['cover'] += stats.get('cover', 0)
                summary_stats['mip gap'] += stats.get('min gap', 0)
                summary_stats['nodes'] += stats.get('nodes', 0)
                summary_stats['time'] += stats.get('time', 0)
                
            except Exception as e:
                print(f"Error occurred in instance {i} with V={V} and alpha={alpha}: {e}")
            
            
        for key in summary_stats:
            if key not in ['V', 'alpha', 'succ']:
                summary_stats[key] = summary_stats[key] / num_instances
                    
        # Ensure 'succ' is an integer
        summary_stats['succ'] = int(summary_stats['succ'])
         
        results.append(summary_stats)
        
        # Store the best solution for this (V, alpha) combination
        if best_instance_index is not None:
            best_sol.append([V, alpha, best_instance_index, best_objective_value, best_percent_ub])
            print(f"Best solution for V={V} and alpha={alpha} is instance {best_instance_index} with objective value {best_objective_value} and %_UB: {best_percent_ub}")

        
# Convert the results into a DataFrame
df_results = pd.DataFrame(results)

# Compute overall average across all V values
overall_average = df_results.groupby('alpha').mean().reset_index()
overall_average['V'] = 'Averages'

# Append overall averages to the results
df_final = pd.concat([df_results, overall_average], ignore_index=True)

# Save to CSV if needed
df_final.to_csv('experiment_results.csv', index=False, sep=',')

# Alternatively, print the DataFrame directly
print(df_final)

# Print the best solutions for each (V, alpha) combination
print("Best solutions across all V and alpha values:")
for best in best_sol:
    print(f"V={best[0]}, alpha={best[1]}, best_instance_index={best[2]}, objective_value={best[3]}, %_UB={best[4]}")

#%% Excel

df_final.to_excel('experiment_results.xlsx', index=False)

#%% Components

#sol = solution.get_value_dict(x)
#x_solution = solution.get_value_dict(x)
#components = [[7,1,6,9,5,8],[2,4,10,3]]

#for component in components: 
#    mdl.add_constraint(mdl.sum(x[i,j] for (i,j) in get_cutset(component,E))>=2)
    
#solution = mdl.solve(log_output = True)
#print(solution)
#%% Connected Components


#%%

# # Initial solve to get the solution
# solution = mdl.solve(log_output=True)
# component_list = find_connected_components(solution, E, V)
# while len(component_list) > 1:
#     for component in component_list: 
#         mdl.add_constraint(mdl.sum(x[i,j] for (i,j) in get_cutset(component, E)) >= 2)
        
#     solution = mdl.solve(log_output=True)
    
#     # Find the connected components
#     component_list = find_connected_components(solution, E, V)

# # Add the constraints for each component
 


#%% Output

