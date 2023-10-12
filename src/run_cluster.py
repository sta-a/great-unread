# %%
%load_ext autoreload
%autoreload 2


# %matplotlib inline


import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import networkx as nx
import networkit as nk
import itertools
import sys
sys.path.append("..")
from utils import DataHandler, TextsByAuthor
from sklearn.pipeline import Pipeline
import itertools

from cluster.create import D2vDist, Delta
from cluster.network import NXNetwork
from cluster.cluster import SimmxCluster
from cluster.sparsifier import Sparsifier

import logging
logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
ticker_logger = logging.getLogger('matplotlib.ticker')
ticker_logger.setLevel(logging.WARNING)
# Disable propagation to root logger
logging.getLogger().setLevel(logging.WARNING)



# # %%
class SimilarityNetwork(DataHandler):
    def __init__(self, language):
        super().__init__(language=language, output_dir='similarity', data_type='csv')
        self.mxs = self.load_mxs()

    def load_mxs(self):
        # # Delta distance mxs
        # delta = Delta(self.language)
        # # delta.create_all_data(use_kwargs_for_fn='mode')
        # all_delta = delta.load_all_data(use_kwargs_for_fn='mode')


        # # D2v distance mxs
        # d2v = D2vDist(language=self.language)
        # all_d2v = d2v.load_all_data(use_kwargs_for_fn='mode', file_string=d2v.file_string)
        # mxs = {**all_delta, **all_d2v}
        # for k, v in mxs.items():
        #     print(k, v)

        # D2v distance mxs
        d2v = D2vDist(language=self.language)
        d2v.modes = ['both']
        all_d2v = d2v.load_all_data(use_kwargs_for_fn='mode', file_string=d2v.file_string)
        mxs = {**all_d2v}
        for k, v in mxs.items():
            print(k, v)

        return mxs


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
        

sn = SimilarityNetwork(language='eng')# .simmx_clustering()


# Heatmap:
# Create a heatmap of the similarity matrix to visualize the pairwise similarities between data points.
# Reorder the rows and columns of the similarity matrix based on the cluster assignments to observe the block-like structures representing different clusters.

# # # Similarity Graphs (Luxburg2007)
# # eta-neighborhodd graph
# # # find eta
# # eta = 0.1
# set all values below eta to 0




# %%
