# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:04:46 2024

@author: jinxu
"""


#test the average competitive ratio of the heuristic approach

import random
import numpy as np
import networkx as nx 
import itertools
import copy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import aoi_utils

N = 1000

naive_cpp_aoi = np.zeros(N)
heu_cpp_aoi = np.zeros(N)
random_cpp_aoi = np.zeros(N)

naive_dup_aoi = np.zeros(N)
heu_dup_aoi = np.zeros(N)
random_dup_aoi = np.zeros(N)
tsp_aoi = np.zeros(N)

case = 0
n_edges = 0
# for ite in range(N):
while True:
    if case >= N:
        break
    
    dice = np.random.rand()
    if dice < 1:
        g = nx.erdos_renyi_graph(10, 0.5, seed=None, directed=False)
    else:
        g = nx.random_labeled_tree(10)
        
    n_edges += g.number_of_edges()
    # else:
        # g = nx.random_lobster(5,0.5,0.5,seed=None)
    # g_array = np.array([[0,1,1,1,1],
    #           [1,0,0,0,0],
    #           [1,0,0,0,0],
    #           [1,0,0,0,0],
    #           [1,0,0,0,0]])
    # g = nx.from_numpy_array(g_array)
    # g = nx.MultiGraph(g)
    if nx.is_planar(g) ==  True and nx.is_eulerian(g) == False and nx.is_connected(g) == True:
    # if nx.is_eulerian(g) == False and nx.is_connected(g) == True:
        # nx.draw_networkx(g, with_labels=True) 
        if case%100 == 0:
            print(case)

    # g_array =     np.array([[0, 4, 0, 0, 0, 0, 0, 8, 0], 
    #                         [4, 0, 8, 0, 0, 0, 0, 11, 0], 
    #                         [0, 8, 0, 7, 0, 4, 0, 0, 2], 
    #                         [0, 0, 7, 0, 9, 0, 14, 0, 0], 
    #                         [0, 0, 0, 9, 0, 10, 0, 0, 0], 
    #                         [0, 0, 4, 0, 10, 0, 2, 0, 0], 
    #                         [0, 0, 0, 14, 0, 2, 0, 1, 6], 
    #                         [8, 11, 0, 0, 0, 0, 1, 0, 7], 
    #                         [0, 0, 2, 0, 0, 0, 6, 7, 0] 
    #                     ]); 
        
    # g = nx.from_numpy_array(g_array)
    # g = nx.MultiGraph(g)
    
        for (u,v,w) in g.edges(data=True):
            w['weight'] = np.random.rand()*10
            # w['weight'] = np.random.lognormal(0,1)
            # w['weight'] = np.random.exponential(1)
        #randomly generate the weight of edges
        g_array = nx.to_numpy_array(g)    
        
        # nx.draw_networkx(g, with_labels=True) 
        g_aug = aoi_utils.smallest_eularian_graph(g)
        source = 0
        
        lower_bound = sum(nx.get_edge_attributes(g, 'weight').values())**2/2
        # naive_circuit = [u for u,v in nx.eulerian_circuit(g_aug, source)] + [source]
        naive_circuit = aoi_utils.deter_eulerian_circuit(g_aug, source)
        heuristic_circuit = aoi_utils.heuristic_AoI_eulerian_circuit(g_aug, source)
        random_eulerian_circuit = aoi_utils.random_eulerian_circuit(g_aug, source)
        #circuit on the CPP graph
        
        g_dup = aoi_utils.add_augmenting_path_to_graph(g, g.edges())
        # naive_circuit_dup = [u for u,v in nx.eulerian_circuit(g_dup, source)] + [source]
        naive_circuit_dup = aoi_utils.deter_eulerian_circuit(g_dup, source)
        heuristic_circuit_dup = aoi_utils.heuristic_AoI_eulerian_circuit(g_dup, source)
        random_eulerian_circuit_dup = aoi_utils.random_eulerian_circuit(g_dup, source)
        #circuits on the duplicated graph
        
        naive_cpp_aoi[case] += aoi_utils.AoI_Compute(g_array,naive_circuit)/lower_bound
        heu_cpp_aoi[case] = aoi_utils.AoI_Compute(g_array,heuristic_circuit)/lower_bound
        random_cpp_aoi[case] = aoi_utils.AoI_Compute(g_array,random_eulerian_circuit)/lower_bound
        
        naive_dup_aoi[case] = aoi_utils.AoI_Compute(g_array,naive_circuit_dup)/lower_bound
        heu_dup_aoi[case] = aoi_utils.AoI_Compute(g_array,heuristic_circuit_dup)/lower_bound
        random_dup_aoi[case] = aoi_utils.AoI_Compute(g_array,random_eulerian_circuit_dup)/lower_bound

        tsp_circuit = aoi_utils.tsp_circuit(g)
        tsp_aoi[case] = aoi_utils.AoI_Compute(g_array,tsp_circuit)/lower_bound

        case += 1
        


print('naive_cpp_aoi',np.mean(naive_cpp_aoi))
print('heu_cpp_aoi',np.mean(heu_cpp_aoi))
print('random_cpp_aoi',np.mean(random_cpp_aoi))

print('Improvement1', 1 - np.mean(heu_cpp_aoi)/np.mean(naive_cpp_aoi))

print('naive_dup_aoi', np.mean(naive_dup_aoi))
print('heu_dup_aoi',np.mean(heu_dup_aoi))
print('random_dup_aoi',np.mean(random_dup_aoi))

print('Improvement2', 1 - np.mean(heu_dup_aoi)/np.mean(naive_dup_aoi))

print(n_edges/N)
# aug_graph = ("CPP", "Duplicated")
# circuit_means = {
#     'Naive': (naive_cpp_aoi,naive_dup_aoi),
#     'Random': (random_cpp_aoi, random_dup_aoi),
#     'Heuristic': (heu_cpp_aoi,heu_dup_aoi),
# }
# max_value = np.max([naive_cpp_aoi, heu_cpp_aoi, random_cpp_aoi, naive_dup_aoi, heu_dup_aoi, random_dup_aoi])
# x = np.arange(len(aug_graph))  # the label locations
# width = 0.2  # the width of the bars
# multiplier = 0

# fig, ax = plt.subplots(layout='constrained')

# for attribute, measurement in circuit_means.items():
#     offset = width * multiplier
#     rects = ax.bar(x + offset, measurement, width, label=attribute)
#     ax.bar_label(rects, padding=2)
#     multiplier += 1
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('AoI')
# # ax.set_title('Penguin attributes by species')
# ax.set_xticks(x + width, aug_graph)
# ax.legend(loc='upper left')
# ax.set_ylim(0, 1.2*max_value)
# plt.show()
# plt.tight_layout()
# fig.savefig('Figure_6_3.pdf', dpi=800, format='pdf')

# matplotlib.rcParams.update({'font.size': 12})
# fig, ax = plt.subplots(figsize=(9, 6))
# # rectangular box plot
# all_data = [naive_cpp_aoi, random_cpp_aoi, heu_cpp_aoi, naive_dup_aoi,  random_dup_aoi, heu_dup_aoi]
# labels = ['naive_cpp','random_cpp','heuristic_cpp','naive_dup','random_dup','heuristic_dup']
# bplot1 = ax.boxplot(all_data,
#                      vert=True,  # vertical box alignment
#                      patch_artist=True,  # fill with color
#                      labels=labels,
#                      showfliers= True,
#                      showmeans=False)
# plt.setp(bplot1['fliers'], markersize=3.0,marker='*')
# colors = ['lightpink', 'lightblue', 'lightgreen','lightpink', 'lightblue', 'lightgreen']
# for l,box in zip(range(6),bplot1['boxes']):
#     # change outline color
#     box.set(color=colors[l], linewidth=2)
#     # change fill color
# # will be used to label x-ticks
# ax.set_ylabel('Ratio')

# plt.show()
# plt.tight_layout()
# # fig.savefig('Figure_6_3.pdf', dpi=800, format='pdf')

matplotlib.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(9, 6))
# rectangular box plot
all_data = [random_cpp_aoi, heu_cpp_aoi, random_dup_aoi, heu_dup_aoi, tsp_aoi]
labels = ['rand_cpp','heu_cpp','rand_dup','heu_dup', 'tsp_heu']
bplot1 = ax.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels,
                     showfliers= True,
                     showmeans=False,
                     medianprops=dict(linewidth=3))
plt.setp(bplot1['fliers'], markersize=3.0,marker='*')
colors = ['lightblue', 'lightgreen', 'lightpink', 'slategrey','orange']
for l,box in zip(range(5),bplot1['boxes']):
    # change outline color
    box.set(color=colors[l], linewidth=2)
    # change fill color
# will be used to label x-ticks
ax.set_ylabel('Ratio')

plt.show()
plt.tight_layout()
fig.savefig('Figure_6_3_new.pdf', dpi=800, format='pdf')