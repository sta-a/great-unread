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
from distance.distance_create import Distance, PydeltaDist, D2vDist
from distance.network import NXNetwork
from distance.cluster import SimmxCluster
import sys
sys.path.append("..")
from utils import DataHandler, TextsByAuthor
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

# a = TextsByAuthor('eng')
# print(a.output_dir)
# # %%
# print(a.nr_texts_per_author)
# # %%
class SimilarityNetwork(DataHandler):
    def __init__(self, language):
        super().__init__(language=language, output_dir='distance', data_type='csv')
        self.mxs = self.load_mxs()

    def set_diagonal(self, mx, value):
        '''
        mx: distance or similarity matrix
        value: the value to set the diagonal to
        '''
        #df.values[[np.arange(df.shape[0])]*2] = value
        for i in range(0, mx.shape[0]):
            mx.iloc[i, i] = value
        return mx

    def load_mxs(self):
        pydelta = PydeltaDist(self.language)
        pydelta.modes = ['burrows-20']##############################################
        pdmxs = pydelta.load_all_data()
        # dd = D2vDist(self.language)
        # #dd.modes = ['doc'] ##############################################
        # dvmxs = dd.load_all_data()
        # #mxs = {**pdmxs}#, **dvmxs}
        # mxs = {**dvmxs}

        # mxs = []
        # for file_name in os.listdir(self.output_dir):
        #     file_path = self.get_file_path(file_name=file_name)
        #     if os.path.isfile(file_path) and file_name.endswith('.csv'):
        #         file_name = os.path.splitext(file_name)[0]  # Extract file name without extension

        #         mx = pd.read_csv(file_path, header=0, index_col=0)
        #         # mx = self.set_diagonal(mx, np.nan)#################3
        #         # mx = mx.iloc[:50, :50] ####################3
        #         # Process the data as needed
        #         mxs.append((file_name, mx))            
        # mxs = mxs
        # return mxs


    def network_clustering(self):
        sparsifier = Sparsifier()
        spars_modes = sparsifier.modes
        spars_parameters = sparsifier.threshold_params
        network = NXNetwork(self.language)

        # Iterate over all combinations and run the steps
        combinations = list(itertools.product(self.mxs, spars_modes, spars_parameters, network.network_cluster_algs, network.attribute_names))
        combinations = list(itertools.product(self.mxs, ['threshold'], spars_parameters, ['gn'], ['gender']))
        for mx_tuple, spars_mode, spars_parameter, cluster_alg, attribute_name in combinations:
            print('------------------------')
            print(mx_tuple[0], spars_mode, spars_parameter, cluster_alg, attribute_name)

            sparsifier = Sparsifier(self.language, mx_tuple[1], spars_mode, spars_parameter)
            spars_mx = sparsifier.sparsify()

            network = NXNetwork(self.language, name_mx_tup=(f'{mx_tuple[0]}-{spars_mode}', spars_mx), cluster_alg=cluster_alg, attribute_name=attribute_name)
            network.get_clusters()


    def simmx_clustering(self):
        network = NXNetwork(self.language)

        # Iterate over all combinations and run the steps
        combinations = list(itertools.product(self.mxs, network.simmx_cluster_algs, network.attribute_names))
        combinations = list(itertools.product(self.mxs, ['spectral'], ['author']))
        for mx_tuple, cluster_alg, attribute_name in combinations:
            print('------------------------')
            print(mx_tuple[0], cluster_alg, attribute_name )

            # Create clusters using the network
            network = NXNetwork(self.language, name_mx_tup=mx_tuple, cluster_alg=cluster_alg, attribute_name=attribute_name)
            network.get_clusters()
        

sn = SimilarityNetwork(language='eng') #.simmx_clustering()


# Heatmap:
# Create a heatmap of the similarity matrix to visualize the pairwise similarities between data points.
# Reorder the rows and columns of the similarity matrix based on the cluster assignments to observe the block-like structures representing different clusters.

# # # Similarity Graphs (Luxburg2007)
# # eta-neighborhodd graph
# # # find eta
# # eta = 0.1
# set all values below eta to 0


# %%
