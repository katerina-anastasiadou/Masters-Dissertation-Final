# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:41:46 2024

@author: katan
"""
import igraph as ig
import matplotlib.pyplot as plt
import random
import math
import numpy as np
from scipy.spatial.distance import euclidean
from itertools import combinations


def get_edges(S,E):
    edges = [] 
    for (i,j) in E:
        #print (i,j)
        if i in S and j in S:
            edges.append((i,j))
    return edges

def map_edges(vertices,edges):
    # vertex_map = list(range(len(vertices)))
    edges_map = []
    for (i,j) in edges:
        edges_map.append((vertices.index(i),vertices.index(j)))
    return edges_map

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

def plot_sol(data,model,V,alpha,instance):
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
    
    # Add a title with V, alpha, and instance
    plt.title(f'Graph for V={V}, alpha={alpha}, instance={instance}')
    
    fig = plt.gcf()
    fig.set_size_inches(8, 8)

def generate_instance(V, alpha):
    
    np.random.seed(None) 
    
    # Step 1: Generate V vertices with coordinates in [0, 1000] x [0, 1000]
    vertices = np.random.randint(0, 1001, size=(V, 2))
    
    # Step 2: Calculate Euclidean distances and round up to the nearest integer
    l_matrix = np.zeros((V, V), dtype=int)
    for i, j in combinations(range(V), 2):
        distance = np.ceil(euclidean(vertices[i], vertices[j]))
        l_matrix[i][j] = l_matrix[j][i] = int(distance)
    
    # Step 3: Define costs
    c_matrix = np.ceil(alpha * l_matrix)  # cij = ⌈α * lij⌉
    d_matrix = np.ceil((10 - alpha) * l_matrix)  # dij = ⌈(10 - α) * lij⌉
    np.fill_diagonal(d_matrix, 0)  # dii = 0 for all i in V
    
    return vertices, l_matrix, c_matrix, d_matrix
