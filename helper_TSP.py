# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:30:04 2024

@author: katan
"""

import igraph as ig
import matplotlib.pyplot as plt


def get_edges(S,E):
    edges = [] 
    for (i,j) in E:
        #print (i,j)
        if i in S and j in S:
            edges.append((i,j))
    return edges

def get_cutset(S,E):
    edges = []
    for (i,j) in E:
        if i in S and j not in S or i not in S and j in S:
            edges.append((i,j))
    return edges

def find_connected_components(model, solution, data):
    """
    Find connected components from the solution of the TSP model.
    
    Args:
    solution (docplex.mp.solution.Solution): The solution of the TSP model.
    E (list of tuples): List of edges.
    V (list): List of vertices.
    
    Returns:
    list of lists: Each sublist contains the nodes in one connected component.
    """
    # Build the graph from the solution
    sol = solution.get_value_dict(model.x)
    edges_in_solution = [(i, j) for (i, j) in data.E if sol[(i, j)] > 0.9]
    
    # Adjust vertex IDs to start from 0
    adjusted_edges = [(i, j) for (i, j) in edges_in_solution]
   
    # Create an igraph graph
    g = ig.Graph()
    g.add_vertices(len(data.V))  # Adding the number of vertices
    g.add_edges(adjusted_edges)
     
    # Get the connected components
    components = g.connected_components(mode='weak')
    
    # Extract the component membership list and adjust back to original IDs
    component_list = [[v for v in comp] for comp in components]
    
    return component_list

def plot_sol(data,model):
    plt.figure()
    for i in data.V:
        plt.scatter(data.loc[i][0],data.loc[i][1], c = 'black')
        plt.annotate(i, (data.loc[i][0]+2,data.loc[i][1]))
    for (i,j) in data.E:
        if model.x[i,j].solution_value > 0.9:
            plt.plot([data.loc[i][0], data.loc[j][0]], [data.loc[i][1], data.loc[j][1]], c = 'blue')
    for i in data.V:
        for j in data.V:
            if model.y[i,j].solution_value > 0.9:
                plt.plot([data.loc[i][0], data.loc[j][0]], [data.loc[i][1], data.loc[j][1]], c = 'red')
            
    plt.axis([0, data.width, 0, data.width])
    plt.grid()
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
