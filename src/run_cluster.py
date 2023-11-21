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
from copy import deepcopy
import itertools

from cluster.create import D2vDist, Delta
from cluster.network import NXNetwork
from cluster.cluster import SimmxCluster, NetworkCluster
from cluster.evaluate import ExtEval, MxIntEval, NkIntEval
from cluster.visualize import MxReorder, NkViz, MxViz
from cluster.sparsifier import Sparsifier
from cluster.cluster_utils import CombinationInfo, MetadataHandler

import logging
logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
# ticker_logger = logging.getLogger('matplotlib.ticker')
# ticker_logger.setLevel(logging.WARNING)
# # Disable propagation to root logger
# logging.getLogger().setLevel(logging.WARNING)



class SimilarityClustering(DataHandler):
    ATTRS = ['gender', 'author', 'canon', 'year', 'features']

    def __init__(self, language):
        super().__init__(language=language, output_dir='similarity', data_type='csv')
        self.mxs = self.load_mxs()
        print(self.mxs)

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

        mxs_list = []
        for name, mx in mxs.items():
            mx.name = name
            mxs_list.append(mx)
        return mxs_list


    def simmx_clustering(self):
        for mx in self.mxs:
            for attr in self.ATTRS:
                for order in MxReorder.ORDERS:
                # for order in ['olo']:
                    info = CombinationInfo(mxname=mx.name, attr=attr, order=order)
                    print(info.as_string())
                    viz = MxViz(self.language, mx, info)
                    if attr != 'features':
                        viz.load_metadata()
                        viz.visualize(plttitle='attrviz')
                    # else:



        # # for mx, cluster_alg in itertools.product(self.mxs, SimmxCluster.ALGS.keys()):
        # for mx, cluster_alg in itertools.product(self.mxs, ['hierarchical']):
        #     sc = SimmxCluster(self.language, cluster_alg, mx)
        #     param_combs = sc.get_param_combinations()

        #     for param_comb in param_combs:
        #         print(param_comb)
        #         clusters = sc.cluster(**param_comb)
        #         inteval = MxIntEval(mx, clusters).evaluate()

        #         # for order in MxReorder.ORDERS:
        #         for order in ['olo']:
        #             info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, attr='cluster', order=order, param_comb=param_comb)
        #             viz = MxViz(self.language, mx, info)
        #             ee = ExtEval(self.language, 'mx', viz, clusters, info, param_comb, inteval)

        #             # for attr in self.ATTRS:
        #             for attr in ['gender']:
        #                     info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, attr=attr, param_comb=param_comb)
        #                     ee.evaluate(info)




    def network_clustering(self):
        for mx, spars_mode, cluster_alg in itertools.product(self.mxs, Sparsifier.MODES.keys(), NetworkCluster.ALGS):
        # for mx, spars_mode, cluster_alg in itertools.product(self.mxs, ['threshold'], ['louvain']):
            spars_params = Sparsifier.MODES[spars_mode]
            for spars_param in spars_params:
                sparsifier = Sparsifier(self.language, mx, spars_mode, spars_param)
                mx = sparsifier.sparsify()

                network = NXNetwork(self.language, mx=mx)

                nc = NetworkCluster(self.language, cluster_alg, network)
                param_combs = nc.get_param_combinations()
                for param_comb in param_combs:
                    clusters = nc.cluster(**param_comb)
                    inteval = NkIntEval(network, clusters, cluster_alg, param_comb).evaluate()
                    for prog in NkViz.PROGS:
                        for attr in self.ATTRS + ['clust']:
                        # for attr in ['gender']:
                            info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, prog=prog, attr=attr, param_comb=param_comb)
                            viz = NkViz(self.language, network, info)
                            ee = ExtEval(self.language, 'nk', viz, clusters, info, param_comb, inteval)
                            ee.evaluate()

import time
mh = MetadataHandler('eng')
s = time.time()
x = mh.get_metadata()
print(f'{time.time()-s} s')
# sn = SimilarityClustering(language='eng').network_clustering()
# sn = SimilarityClustering(language='eng').simmx_clustering()


# Attr in networkviz
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
