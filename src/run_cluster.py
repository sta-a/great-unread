# %%
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import itertools
import sys
sys.path.append("..")
from utils import DataHandler
import itertools

from cluster.create import D2vDist, Delta
from cluster.network import NXNetwork
from cluster.cluster import SimmxCluster, NetworkCluster, ExtEval, IntEval, MxReorder
from cluster.sparsifier import Sparsifier
from cluster.cluster_utils import CombinationInfo

import logging
logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
ticker_logger = logging.getLogger('matplotlib.ticker')
ticker_logger.setLevel(logging.WARNING)
# Disable propagation to root logger
logging.getLogger().setLevel(logging.WARNING)



class SimilarityClustering(DataHandler):
    ATTRS = ['gender', 'author', 'canon', 'year', 'features']
    
    def __init__(self, language):
        super().__init__(language=language, output_dir='similarity', data_type='csv')
        self.mxs = self.load_mxs()

    def load_mxs(self):
        full = False
        if full:
            # Delta distance mxs
            delta = Delta(self.language)
            # delta.create_all_data(use_kwargs_for_fn='mode')
            all_delta = delta.load_all_data(use_kwargs_for_fn='mode', subdir=True)


            # D2v distance mxs
            d2v = D2vDist(language=self.language)
            all_d2v = d2v.load_all_data(use_kwargs_for_fn='mode', file_string=d2v.file_string, subdir=True)
            mxs = {**all_delta, **all_d2v}
            for k, v in mxs.items():
                print(k, v)

        else:
            # D2v distance mxs
            d2v = D2vDist(language=self.language)
            d2v.modes = ['both']
            all_d2v = d2v.load_all_data(use_kwargs_for_fn='mode', file_string=d2v.file_string, subdir=True)
            mxs = {**all_d2v}
        return mxs


    def network_clustering(self):
        # for mx, spars_mode, cluster_alg in itertools.product(self.mxs.values(), Sparsifier.MODES.keys(), NetworkCluster.ALGS):
        for mx, spars_mode, cluster_alg in itertools.product(self.mxs.values(), ['threshold'], ['gn']):
            print('------------------------', mx.name, spars_mode, cluster_alg)
            spars_params = Sparsifier.MODES[spars_mode]
            for spars_param in spars_params:
                sparsifier = Sparsifier(self.language, mx, spars_mode, spars_param)
                mx = sparsifier.sparsify()

                network = NXNetwork(self.language, mx=mx, cluster_alg=cluster_alg)
                cl = network.get_clusters()
                print(cl)

                # # for attr in self.ATTRS:
                # for attr in ['gender']: #, 'author', 'canon', 'year']:
                #     info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, attr=attr, param_comb=param_comb)



    # def simmx_clustering(self):
    #     for mx, cluster_alg in itertools.product(self.mxs.values(), SimmxCluster.ALGS.keys()):
    #         print('------------------------')

    #         sc = SimmxCluster(self.language, mx, cluster_alg)
    #         param_combs = sc.get_param_combinations()
    #         for param_comb in param_combs:
    #             clusters = sc.cluster(**param_comb)
    #             inteval = IntEval(mx, clusters, param_comb).evaluate()

    #             for attr in self.ATTRS:
    #              £   for order in MxReorder.ORDERS:
    #                     info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, attr=attr, order=order, param_comb=param_comb)
    #                     ce = ExtEval(self.language, mx, clusters, info, inteval, param_comb)
    #                     ce.evaluate()


    def simmx_clustering(self):
        for mx, cluster_alg in itertools.product(self.mxs.values(), ['hierarchical']):
            sc = SimmxCluster(self.language, mx, cluster_alg)
            param_combs = sc.get_param_combinations()
            print(param_combs)
            for param_comb in param_combs:
                clusters = sc.cluster(**param_comb)
                inteval = IntEval(mx, clusters, param_comb).evaluate()

                for attr in ['gender']: #'gender', 'author', 'canon', 'year'
                    for order in ['olo']:
                        info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, attr=attr, order=order, param_comb=param_comb)
                        print(info.as_df())
                        ce = ExtEval(self.language, mx, clusters, info, inteval, param_comb)
                        ce.evaluate()


        

sn = SimilarityClustering(language='eng').simmx_clustering()


# Elbow for internal cluster evaluation



# Hierarchical clustering:
# From scipy documentation, ignored here: Methods ‘centroid’, ‘median’, and ‘ward’ are correctly defined only if Euclidean pairwise metric is used. If y is passed as precomputed pairwise distances, then it is the user’s responsibility to assure that these distances are in fact Euclidean, otherwise the produced result will be incorrect.

# Heatmap:
# Create a heatmap of the similarity matrix to visualize the pairwise similarities between data points.
# Reorder the rows and columns of the similarity matrix based on the cluster assignments to observe the block-like structures representing different clusters.

# # # Similarity Graphs (Luxburg2007)
# # eta-neighborhodd graph
# # # find eta
# # eta = 0.1
# set all values below eta to 0


# %%
