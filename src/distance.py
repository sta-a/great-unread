# %%
%load_ext autoreload
%autoreload 2
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import numpy as np
from distance_create import load_distance_mx
import networkx as nx
import networkit as nk
from distance_analysis import get_mx_triangular, nr_elements_triangular, distance_to_similarity_mx
from distance_sparsify import filter_min_author_similarity, filter_threshold
from distance_visualization import plot_distance_distribution
from network_functions import check_symmetric, nx_print_graph_info, nx_graph_from_mx, nx_plot_graph

data_dir = '../data'


### Save plots doesnt work #############################3

  

for language in ['eng']: #, 'ger'
    distances_dir = os.path.join(data_dir, 'distances', language)
    if not os.path.exists(distances_dir):
        os.makedirs(distances_dir, exist_ok=True)
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    canonscores_dir = os.path.join(data_dir, 'canonscores')
    features_dir = os.path.join(data_dir, 'features_None', language)


    #Jannidis2015
    # 'edersimple' does not work because corpus.sqrt() doesn't exist, definition unclear
    # In R also called 'Eder’s Simple Distance)'
    # pydelta_dists = ['burrows', 'cosinedelta', 'quadratic', 'eder'] #######
    # nmfw_list = [500, 1000, 2000, 5000]
    pydelta_dists = ['burrows']
    nmfw_list = [500]
    for dist in pydelta_dists:
        for nmfw in nmfw_list:
            print('----------------------------', dist, nmfw)
            # Get distance matrix
            dist_name = f'{dist}{nmfw}'
            dmx = load_distance_mx(language, data_dir, dist_name=dist_name, nmfw=nmfw, function=dist)
            assert dmx.index.equals(dmx.columns) # Check if rows and cols are sorted'
            # dmx = dmx.iloc[:50,:50]
            # plot_distance_distribution(mx=dmx, mx_type='distance', language=language, filename=dist_name, data_dir=data_dir)

            # Turn into similarity matrix, diagonal is Nan
            smx = distance_to_similarity_mx(dmx)
            assert smx.index.equals(smx.columns) # Check if rows and cols are sorted
            assert not np.any(smx.values == 0) # Test whether any element is 0

            # plot_distance_distribution(mx=smx, mx_type='similarity', language=language, filename=dist_name, data_dir=data_dir)
            smx = smx.iloc[:50,:50]

            # directed_mx = filter_min_author_similarity(smx)

            threshold = 0.8
            tsmx = filter_threshold(mx=smx, q=threshold)
            print(f'Nr expected edges in filtered graph: {nr_elements_triangular(smx)*(1-threshold)}') # Only values above threshold are left

            # edge_labels = nx.get_edge_attributes(tsmxG,'weight')
            # for i,v in edge_labels.items():
            #     print(i,v

            tsmxG = nx_graph_from_mx(mx=tsmx)
            nx_print_graph_info(tsmxG)

            louvain_c = nx.community.louvain_communities(tsmxG, weight='weight', seed=11, resolution=0.1)

            nx_plot_graph(tsmxG, cluster_type='louvain', cluster_assignments=louvain_c)


# # Similarity Graphs (Luxburg2007)
# eta-neighborhodd graph
# # find eta
# eta = 0.1
# set all values below eta to 0

# %%
