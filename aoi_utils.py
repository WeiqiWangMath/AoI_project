#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:43:31 2024

@author: jin
"""

import random
import numpy as np
import networkx as nx 
import itertools
import copy
import pandas as pd
import matplotlib.pyplot as plt

def sum_edges(graph):
    """Calculate the sum of all edges in a nx graph, including repeat edges"""
    for (u,v,w) in graph.edges(data=True):
        # print(u,v,w)
        if w == None:
            w = 0

    edge_weight_sum = np.sum(weight for (u, v, weight) in graph.edges.data('weight'))
    return edge_weight_sum

def smallest_eularian_graph(graph):
    """
    Add edges with the smallest weight sum to the input graph to create the smallest Eulerian graph. 
    An Eulerian route of the output graph is a Chinese Postman Problem solution to the original graph.
    Input: original graph
    Output: an Eulerian graph contains all edges in original graph
    """
    # Calculate list of nodes with odd degree
    nodes_odd_degree = [v for v, d in graph.degree() if d % 2 == 1]
    odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))
    l = (len(odd_node_pairs)+1)//2

    # Compute the shortest distance between each pair of nodes in a graph.  Return a dictionary keyed on node pairs (tuples)."""
    distances = {}
    for pair in odd_node_pairs:
        distances[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1], weight='weight')
    # print(distances)
    # Create a completely connected graph using a list of vertex pairs and the shortest path distances between them
    gf = nx.Graph()
    for k, v in distances.items():
        gf.add_edge(k[0], k[1], weight= -v)  
    
    # Compute min weight matching.
    # Note: max_weight_matching uses the 'weight' attribute by default as the attribute to maximize.
    odd_matching_dupes = nx.algorithms.max_weight_matching(gf, True)
    # odd_matching_dupes = nx.algorithms.min_weight_matching(gf, True)

    # print(odd_matching_dupes)
    # Convert matching to list of deduped tuples
    odd_matching = list(pd.unique([tuple(sorted([k, v])) for k, v in odd_matching_dupes]))
    # print(odd_matching)
    # Create a new graph to overlay on g_odd_complete with just the edges from the min weight matching
    g_odd_complete_min_edges = nx.Graph(odd_matching)
    # print(odd_matching)
    graph_aug = nx.MultiGraph(graph.copy())
    # aug_edge = []
    # for en, e in enumerate(odd_matching):
    #     aug_path = nx.shortest_path(graph, e[0], e[1])
    #     # print(e)
    #     for edge_pair in list(zip(aug_path[:-1], aug_path[1:])):
    #         # print(edge_pair)
    #         graph_aug.add_edge(edge_pair[0], 
    #                         edge_pair[1], 
    #                         weight = graph.get_edge_data(edge_pair[0],edge_pair[1])['weight'])
    # Create augmented graph: add the min weight matching edges to g
    g_aug = add_augmenting_path_to_graph(graph, odd_matching)
    
    return g_aug
    # return graph_aug


def add_augmenting_path_to_graph(graph, min_weight_pairs):
    """
    Add the minimum weight matching edges to the original graph
    Parameters:
        graph: NetworkX graph (original graph)
        min_weight_pairs: list[tuples] of node pairs from min weight matching
    Returns:
        augmented NetworkX graph
    """
    
    # We need to make the augmented graph a MultiGraph so we can add parallel edges
    graph_aug = nx.MultiGraph(graph.copy())
    for pair in min_weight_pairs:
        aug_path = nx.shortest_path(graph, pair[0], pair[1],weight='weight')
        # print(aug_path)
        for edge_pair in list(zip(aug_path[:-1], aug_path[1:])):
            graph_aug.add_edge(edge_pair[0], 
                            edge_pair[1], 
                            weight = graph[edge_pair[0]][edge_pair[1]]['weight'])
    return graph_aug




def AoI_Compute(A,route):
    """
    Compute the AoI for a given route
    Input: 
    Adjacency matrix A contains the distance between nodes, 0 present no edge between nodes
    route: Patrol route to compute AoI. The route must start and end at the same node.
    Output:
    A real number: Route AoI for the graph 
    """
    weight = np.ones_like(A)  # Uniform weight

    M = len(route)
    N = len(A)
    two_route = route[0:M-1]+ route

    # Find the length of the patrol route
    route_len = 0
    for current_edge in range(M-1):
        route_len += A[route[current_edge]][route[current_edge+1]]
    #print("Patrol route length")
    #print(route_len)


    AOI = 0
    # Traverse all edges in the graph
    for node_1 in range(N):
        for node_2 in range(node_1+1,N):  
            if A[node_1][node_2]>0:
                AOI_current_edge = 0
                
                # Find the edge in the patrol route
                for current_edge in range(M-1,2*M-2):
                    if (node_1 == two_route[current_edge] and node_2 == two_route[current_edge+1]) or \
                    (node_2 == two_route[current_edge] and node_1 == two_route[current_edge+1]):
                        
                        # Find the last apperence of the edge in the patrol route
                        last_edge = current_edge - 1
                        temp_len = 0
                        while last_edge >= 0:
                            if (node_1 == two_route[last_edge] and node_2 == two_route[last_edge+1]) or \
                            (node_2 == two_route[last_edge] and node_1 == two_route[last_edge+1]):
                                break
                            temp_len += A[two_route[last_edge]][two_route[last_edge+1]]
                            last_edge = last_edge - 1
                        AOI_current_edge += 1/2 * temp_len * A[node_1][node_2] * (temp_len + A[node_1][node_2])
    #                    AOI_current_edge += 1/6 * A[node_1][node_2]**3
                        if two_route[last_edge] == two_route[current_edge]:
                            #clockwise
                            AOI_current_edge += 1/2 * A[node_1][node_2]**3 + 1/2 * A[node_1][node_2]**2 * temp_len
                        else:
                            AOI_current_edge += 2/3 * A[node_1][node_2]**3 + 1/2 * A[node_1][node_2]**2 * temp_len
                            
                # If there is edge not in the route, return error
                if AOI_current_edge == 0:
                    print("Error: infinite AOI")
                AOI += AOI_current_edge * weight[node_1][node_2]
    AOI = AOI / route_len
    # print("AOI of the route")
    # print(AOI)
    if np.sum(np.sum(A))/2 > route_len + 0.01:
        print('Error: Edges have not been enumerated')
    return AOI

def is_connected_after_removing_edge(G, edge):
    """
    Check if removing the edge separates the graph into two disconnected sets of edges. (An important step in Fleury's algorithm)
    Input: Graph G and edge
    Output: True: connected after removing edge 
    """
    if G.degree(edge[0]) == 1:
        return True
    
    # Make a copy of the graph to avoid modifying the original
    G_temp = G.copy()
    
    # Remove the specified edge
    G_temp.remove_edge(*edge)
    
    # Check if the graph is still connected
    is_still_connected = nx.is_connected(G_temp)
    
    return is_still_connected

# def heuristic_choice(G, vertex_queue, edges_connect):
#     """
#     The heuristic edge-selecting algorithem in order to minimize AoI
#     Input:
#         G: graph with added edges
#         vertex_queue: Current patrol route presented by an ordered node queue
#         edges_connect: All possible edge (to form an Eulerian graph) at current node.
#     Output:
#         (u, v): The edge selection in heuristic algorithm.
#     """
#     total_len = sum_edges(G)
    
#     # We use the time from the last appearance of the edge to determine which edge to traverse:
#     # 1. If the edge is only patroled once in the graph, then set the time is total_len/2
#     # 2. If the edge (u,v) is patroled twice in the graph, but not in the route, set (the time from last traverse) = (the current time)+ min(dist(source,u), dist(source,v)). If the value < total_len/2, we set the value= total_len/2 + 0.01 to priortize them over edges that only be patroled once.
#     # 3. If the edge (u,v) is patroled twice in the graph and in the route, calculate the time from last traverse.
#     # Then selete the edge with maxinum time from the last appearance to traverse, if more than one edge has the same value, we randomly choose one.
    
#     time_last_app = []
#     for u,v in edges_connect:
#         if G.number_of_edges(u,v) == 1:
#             time_temp = total_len/2+0.01
#         else:
#             time_temp = G.get_edge_data(u, v)[0]['weight']
#             flag_edge_in_route = False
#             for i in range(len(vertex_queue)-1,0,-1):
#                 if (u,v) == (vertex_queue[i],vertex_queue[i-1]) or (u,v) == (vertex_queue[i-1],vertex_queue[i]):
#                     flag_edge_in_route = True
#                     break
#                 time_temp += G.get_edge_data(vertex_queue[i-1], vertex_queue[i])[0]['weight']
#             if not flag_edge_in_route:
#                 time_temp += min(nx.shortest_path_length(G, vertex_queue[0], u), nx.shortest_path_length(G, vertex_queue[0], v))
#                 time_temp = max(total_len/2+0.01, time_temp)
#         time_last_app.append(time_temp)
#         # print(u,v,time_last_app)
#     largest_element = max(time_last_app)
#     indices = [index for index, element in enumerate(time_last_app) if element == largest_element]
#     index_max = random.choice(indices)
#     # print("Route:", vertex_queue)
#     # print(edges_connect, time_last_app)
#     # print(time_last_app,indices)
#     return edges_connect[index_max]
    

# def heuristic_choice(G, vertex_queue, edges_connect):
#     """
#     The heuristic edge-selecting algorithem in order to minimize AoI (Edited, with a small probability of randomly select the edges)
#     Input:
#         G: graph with added edges
#         vertex_queue: Current patrol route presented by an ordered node queue
#         edges_connect: All possible edge (to form an Eulerian graph) at current node.
#     Output:
#         (u, v): The edge selection in heuristic algorithm.
#     """
#     total_len = sum_edges(G)
    
#     # We use the time from the last appearance of the edge to determine which edge to traverse:
#     # 1. If the edge is only patroled once in the graph, then set the time is total_len/2
#     # 2. If the edge (u,v) is patroled twice in the graph, but not in the route, set (the time from last traverse) = (the current time)+ min(dist(source,u), dist(source,v)). If the value < total_len/2, we set the value= total_len/2 + 0.01 to priortize them over edges that only be patroled once.
#     # 3. If the edge (u,v) is patroled twice in the graph and in the route, calculate the time from last traverse.
#     # Then selete the edge with maxinum time from the last appearance to traverse, if more than one edge has the same value, we randomly choose one.
    
#     time_last_app = []
#     rand = np.random.rand()
#     if rand < 0.0001:
#         return random.choice(edges_connect)
#     else:
#         for u,v in edges_connect:            
#             if G.number_of_edges(u,v) == 1:
#                 time_temp = total_len/2 
#             else:
#                 time_temp = G.get_edge_data(u, v)[0]['weight']
#                 flag_edge_in_route = False
#                 for i in range(len(vertex_queue)-1,0,-1):
#                     if (u,v) == (vertex_queue[i],vertex_queue[i-1]) or (u,v) == (vertex_queue[i-1],vertex_queue[i]):
#                         flag_edge_in_route = True
#                         break
#                     time_temp += G.get_edge_data(vertex_queue[i-1], vertex_queue[i])[0]['weight']
#                 if not flag_edge_in_route:
#                     time_temp += min(nx.shortest_path_length(G, vertex_queue[0], u), nx.shortest_path_length(G, vertex_queue[0], v))
#                     time_temp = max(total_len/2+0.1, time_temp)
#             time_last_app.append(time_temp)
#         largest_element = max(time_last_app)
#         indices = [index for index, element in enumerate(time_last_app) if element == largest_element]
#         index_max = random.choice(indices)
#         # print("Route:", vertex_queue)
#         # print(edges_connect, time_last_app)
#         return edges_connect[index_max]

def heuristic_choice(G, vertex_queue, edges_connect):
    """
    The heuristic edge-selecting algorithem in order to minimize AoI (Edited, with a small probability of randomly select the edges)
    Input:
        G: graph with added edges
        vertex_queue: Current patrol route presented by an ordered node queue
        edges_connect: All possible edges (to form an Eulerian graph) at the current node.
    Output:
        (u, v): The edge selection in the heuristic algorithm.
    """
    total_len = sum_edges(G)
    
    # We use the time from the last appearance of the edge to determine which edge to traverse:
    # 1. If the edge is only patrolled once in the graph, then set the time as total_len/2
    # 2. If the edge (u,v) is patrolled twice in the graph but not in the route, set (the time from the last traverse) = (the current time)+ dist(source,v). If the value < total_len/2, we set the value= total_len/2 + 0.01 to priortize them over edges that only be patroled once.
    # 3. If the edge (u,v) is patrolled twice in the graph and in the route, calculate the time from the last traverse.
    # Then, delete the edge with the maximum time from the last appearance to traverse; if more than one edge has the same value, we randomly choose one.
    
    time_last_app = []
    rand = np.random.rand()
    if rand < 0.00001:
        return random.choice(edges_connect)
    else:
        for u,v in edges_connect:            
            if G.number_of_edges(u,v) == 1:
                time_temp = total_len/2 
            else:
                time_temp = G.get_edge_data(u, v)[0]['weight']
                flag_edge_in_route = False
                for i in range(len(vertex_queue)-1,0,-1):
                    if (u,v) == (vertex_queue[i],vertex_queue[i-1]) or (u,v) == (vertex_queue[i-1],vertex_queue[i]):
                        flag_edge_in_route = True
                        break
                    time_temp += G.get_edge_data(vertex_queue[i-1], vertex_queue[i])[0]['weight']
                if not flag_edge_in_route:
                    time_temp += nx.shortest_path_length(G, vertex_queue[0], v)
                    time_temp = max(total_len/2+0.1, time_temp)
            time_last_app.append(time_temp)
        largest_element = max(time_last_app)
        indices = [index for index, element in enumerate(time_last_app) if element == largest_element]
        index_max = random.choice(indices)
        # print("Route:", vertex_queue)
        # print(edges_connect, time_last_app)
        return edges_connect[index_max]


def heuristic_AoI_eulerian_circuit(G, source):
    """
    Modified Fleury's algorithm to heuristically select the route that minimizes AoI for a given Eulerian graph. 
    Input:
        G: An Eulerian graph
        source: source of the route
    Output:
        The patrol route is presented by a queue of vertex. Start and end at the source node.
    """
    G_temp = G.copy()
    vertex_queue = [source]
    current_len = 0
    route_len = sum_edges(G)
    while current_len < route_len - 0.001:
        edges_connect = list(set([(u,v) for u,v in G_temp.edges(vertex_queue[-1]) if is_connected_after_removing_edge(G_temp, (u,v))]))
        # print(edges_connect)
        # print(current_len, route_len, edges_connect,vertex_queue[-1])
        # Apply heuristic algo here
        u,v = heuristic_choice(G, vertex_queue, edges_connect)
        
        vertex_queue.append(v)
        G_temp.remove_edge(u, v)
        if G_temp.degree(u) == 0:
            G_temp.remove_node(u)
        current_len += G.get_edge_data(u, v)[0]['weight']
        
    return vertex_queue

def random_eulerian_circuit(G, source):
    """
    Fleury's algorithm to generate an Eulerian route for a given Eulerian graph. If there are multiple choices when selecting edges, the function randomly selects one of them.
    Input:
        G: An Eulerian graph
        source: source of the route
    Output:
        The patrol route is presented by a queue of vertex. Start and end at the source node.
    """
    G_temp = G.copy()
    vertex_queue = [source]
    current_len = 0
    route_len = sum_edges(G)
    while current_len < route_len - 0.001:
        edges_connect = list(set([(u,v) for u,v in G_temp.edges(vertex_queue[-1]) if is_connected_after_removing_edge(G_temp, (u,v))]))
        
        # Apply random choice here
        u,v = random.choice(edges_connect)
        
        vertex_queue.append(v)
        G_temp.remove_edge(u, v)
        if G_temp.degree(u) == 0:
            G_temp.remove_node(u)
        current_len += G.get_edge_data(u, v)[0]['weight']
        
    return vertex_queue

def deter_eulerian_circuit(G, source):
    """
    Fleury's algorithm to generate Eulerian route for a given Eulerian graph. If there are multiple choice when selecting edges, the function randomly select one of them.
    Input:
        G: An Eulerian graph
        source: source of the route
    Output:
        The patrol route presented by a queue of vertex. Start and end at source node.
    """
    G_temp = G.copy()
    vertex_queue = [source]
    current_len = 0
    route_len = sum_edges(G)
    while current_len < route_len - 0.001:
        edges_connect = list(set([(u,v) for u,v in G_temp.edges(vertex_queue[-1]) if is_connected_after_removing_edge(G_temp, (u,v))]))
        
        # Apply random choice here
        u,v = edges_connect[0]
        
        vertex_queue.append(v)
        G_temp.remove_edge(u, v)
        if G_temp.degree(u) == 0:
            G_temp.remove_node(u)
        current_len += G.get_edge_data(u, v)[0]['weight']
        
    return vertex_queue

def tsp_circuit(G):
    line_g = nx.line_graph(G)
    #transform the original graph into a line graph
    for (u,v,w) in line_g.edges(data=True):
        # w['weight'] = np.random.rand()*10
        w['weight'] = 1.
    tsp_tour = nx.approximation.traveling_salesman_problem(line_g)
    #using the Christofide algorithm to solve the TSP.
    #obtain a route that travels through all the nodes within the line graph
    #each step of the route shows the edge that should be traversed
    
    # nx.draw_networkx(line_g, with_labels=True)
    
    #the following is to reconstruct the circuit on the original graph
    tsp_circuit = []
    temp_node = set(tsp_tour[0]).intersection(tsp_tour[1])
    tsp_circuit.append(set(tsp_tour[0]).difference(tsp_tour[1]).pop())
    tsp_circuit.append(temp_node.pop())   
    #the origin of the circuit is to find the first two elements of the route
    # Example: the route shows that the circuit should travel (0,3) and (3,7) at the beginning
    # we then reconstruct the circuit by finding temp_node = 3, which is the 
    # node that connects two edges and is also next node to be visited
    # the first node in the route is thus (0,3)\{3} = {0}
       
    for i in range(1,len(tsp_tour)-1):
        temp_node = set(tsp_tour[i]).intersection(tsp_tour[i+1]).pop()
        #again, the temp_node is the node that connected the next edge
        #it is supposed to be the out node of the current edge tsp_tour[i]
        temp_node_last = tsp_circuit[-1]
        #this is the current node that we stop at
        if temp_node_last == temp_node:
            tsp_circuit.append(set(tsp_tour[i]).difference(tsp_tour[i+1]).pop())
            tsp_circuit.append(temp_node)
            #if the out node is equal to the current node, it means that we should traverse the edge twice
            #by visiting the other node on the edge first and travel back           
        else:
            tsp_circuit.append(temp_node)
    tsp_circuit.append(tsp_circuit[0])
    if tsp_circuit[-1] == tsp_circuit[-2]:
        del tsp_circuit[-1]
        #this is to avoid the case in which the origin is recorded repeatedly
    return tsp_circuit