# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import networkx as nx
import networkit as nk
import itertools
from distance.distance_create import Distance
from distance.network import NXNetwork
from distance.cluster import DataClusterer
import sys
sys.path.append("..")
from utils import DataHandler
from sklearn.pipeline import Pipeline
import itertools

from distance.sparsifier import Sparsifier

import logging
logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
ticker_logger = logging.getLogger('matplotlib.ticker')
ticker_logger.setLevel(logging.WARNING)
# Disable propagation to root logger
logging.getLogger().setLevel(logging.WARNING)


### Save plots doesnt work #############################3
# 'edersimple' does not work because corpus.sqrt() doesn't exist, definition unclear
# In R also called 'Eder’s Simple Distance)'
# Jannidis2015


# directory = '/home/annina/scripts/great_unread_nlp/data/distance/eng'###########
# if os.path.exists(directory) and os.path.isdir(directory):
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         if os.path.isfile(file_path):
#             os.remove(file_path)



class SimilarityNetwork(DataHandler):
    def __init__(self, language):
        super().__init__(language=language, output_dir='distance', data_type='csv')
        self.mxs= self.load_mxs()

    def load_mxs(self):
        # pydelta = PydeltaDist(self.language)
        # pydelta.modes = ['burrows-500']##############################################
        # pdmxs = pydelta.load_all_data()
        # dd = D2vDist(self.language)
        # #dd.modes = ['doc_tags'] ##############################################
        # dvmxs = dd.load_all_data()
        #mxs = {**pdmxs}#, **dvmxs}
        #mxs = {**dvmxs}

        mxs = []
        for file_name in os.listdir(self.output_dir)[:2]:
            file_path = self.get_file_path(file_name=file_name)
            if os.path.isfile(file_path) and file_name.endswith('.csv'):
                file_name = os.path.splitext(file_name)[0]  # Extract file name without extension

                mx = pd.read_csv(file_path, header=0, index_col=0)
                mx = mx.iloc[:50, :50] ####################3
                # Process the data as needed
                mxs.append((file_name, mx))            
        mxs = mxs
        return mxs


    def run_pipeline(self):
        sparsifier = Sparsifier()
        spars_modes = sparsifier.modes
        spars_parameters = sparsifier.threshold_params
        network = NXNetwork(self.language)

        # Iterate over all combinations and run the steps
        combinations = list(itertools.product(self.mxs, spars_modes, spars_parameters, network.cluster_algs, network.attribute_names))
        combinations = list(itertools.product(self.mxs, ['author'], spars_parameters, network.cluster_algs, network.attribute_names))
        for mx_tuple, spars_mode, spars_parameter, cluster_alg, attribute_name in combinations:
            print('####################################################')
            print(mx_tuple[0], spars_mode, spars_parameter, cluster_alg, attribute_name)

            sparsifier = Sparsifier(mx_tuple[1], spars_mode, spars_parameter)
            spars_mx = sparsifier.sparsify()
            spars_mx.to_csv(f'sparse-{mx_tuple[0]}-authors-{self.language}')

            # Create clusters using the network
            network = NXNetwork(self.language, name_mx_tup=(f'{mx_tuple[0]}-{spars_mode}', spars_mx), cluster_alg=cluster_alg, attribute_name=attribute_name)
            clusters = network.create_clusters()
        

sn = SimilarityNetwork(language='eng').run_pipeline()






# directed_mx = filter_min_author_similarity(smx)

# tsmx = filter_threshold(mx=smx, q=threshold)
# print(f'Nr expected edges1 in filtered graph: {nr_elements_triangular(smx)*(1-threshold)}') # Only values above threshold are left

# edge_labels = nx.get_edge_attri
# butes(tsmxG,'weight')
# for i,v in edge_labels.items():
#     print(i,v

#             # tsmxG = network_from_mx(mx=tsmx)
#             # nx_print_graph_info(tsmxG)
#             # louvain_c = nx.community.louvain_communities(tsmxG, weight='weight', seed=11, resolution=0.1)
#             # nx_plot_graph(tsmxG, cluster_alg='louvain', cluster_list=louvain_c)

#             nv = NetworkViz(
#                 mx = tsmx, 
#                 G = None,
#                 draw = True,
#                 attribute_name = 'canon', # cluster: unlabeled groups
#                 attribute_name = None, # attribute: labeled groups, i.e. 'm', 'f'
#                 distances_dir = distances_dir,
#                 metadata_dir = metadata_dir,
#                 canonscores_dir = canonscores_dir,
#                 language = language)


# # # Similarity Graphs (Luxburg2007)
# # eta-neighborhodd graph
# # # find eta
# # eta = 0.1
# set all values below eta to 0


# %%
