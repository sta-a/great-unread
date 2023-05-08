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
from network_functions import nk_from_adjacency
import networkx as nx
import networkit as nk
from distance_sparsify import distance_to_similarity_mx, filter_min_author_similarity
from distance_visualization import visualize_directed_graph, plot_distance_distribution


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
            # visualize_directed_graph(directed_mx)


            filter_simmelian(smx)





# %%

def filter_simmelian(mx):
    '''
    mx: similarity matrix
    '''

    G = nk_from_adjacency(mx)
    print(nk.overview(G))
    G.indexEdges()
    targetRatio = 0.2
    ## Non-parametric
    simmelianSparsifier = nk.sparsification.SimmelianSparsifierNonParametric()
    simmelieanGraph = simmelianSparsifier.getSparsifiedGraphOfSize(G, targetRatio) # Get sparsified graph
    print('Nr edges before and after filtering', G.numberOfEdges(), simmelieanGraph.numberOfEdges())
    x = simmelianSparsifier.scores(G)
    # same nr edges after sparsification - weights not considered?







# # Similarity Graphs (Luxburg2007)
# eta-neighborhodd graph
# # find eta
# eta = 0.1
# set all values below eta to 0