# %%
# %load_ext autoreload
# %autoreload 2

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
from cluster.evaluate import MxExtEval, MxIntEval, NkExtEval
from cluster.visualize import MxReorder
from cluster.sparsifier import Sparsifier
from cluster.cluster_utils import CombinationInfo

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

        return mxs


    # def simmx_clustering(self):
    #     for mx, cluster_alg in itertools.product(self.mxs.values(), SimmxCluster.ALGS.keys()):
    #         print('------------------------')

    #         sc = SimmxCluster(self.language, cluster_alg, mx)
    #         param_combs = sc.get_param_combinations()
    #         for param_comb in param_combs:
    #             clusters = sc.cluster(**param_comb)
    #             inteval = MxIntEval(mx, clusters, param_comb).evaluate()

    #             for attr in self.ATTRS:
    #                 for order in MxReorder.ORDERS:
    #                     info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, attr=attr, order=order, param_comb=param_comb)
    #                     ce = MxExtEval(self.language, mx, clusters, info, param_comb, inteval)
    #                     ce.evaluate()


    def simmx_clustering(self):
        for mx, cluster_alg in itertools.product(self.mxs.values(), ['hierarchical']):
            sc = SimmxCluster(self.language, cluster_alg, mx)
            param_combs = sc.get_param_combinations()
            print(param_combs)
            for param_comb in param_combs:
                clusters = sc.cluster(**param_comb)
                inteval = MxIntEval(mx, clusters, param_comb).evaluate()

                for attr in ['gender']: #'gender', 'author', 'canon', 'year'
                    for order in ['olo']:
                        info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, attr=attr, order=order, param_comb=param_comb)
                        print(info.as_df())
                        me = MxExtEval(self.language, mx, clusters, info, param_comb, inteval)
                        me.evaluate()




    def network_clustering(self):
        # for mx, spars_mode, cluster_alg in itertools.product(self.mxs.values(), Sparsifier.MODES.keys(), NetworkCluster.ALGS):
        for mx, spars_mode, cluster_alg in itertools.product(self.mxs.values(), ['threshold'], ['louvain']):
            print('------------------------', mx.name, spars_mode, cluster_alg)
            # mx.mx = mx.mx.iloc[:50, :50]
            spars_params = Sparsifier.MODES[spars_mode]
            for spars_param in spars_params:
                sparsifier = Sparsifier(self.language, mx, spars_mode, spars_param)
                mx = sparsifier.sparsify()

                network = NXNetwork(self.language, mx=mx)

                nc = NetworkCluster(self.language, cluster_alg, network)
                param_combs = nc.get_param_combinations()
                print(param_combs)
                for param_comb in param_combs:
                    clusters = nc.cluster(**param_comb)
                
                    # for attr in self.ATTRS:
                    for attr in ['gender']: #, 'author', 'canon', 'year']:
                        info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, attr=attr, param_comb=param_comb)
                        ne = NkExtEval(self.language, network, clusters, info, param_comb)
                        ne.evaluate()
                        print(info.as_df())



sn = SimilarityClustering(language='eng').network_clustering()


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
# class MxViz(DataHandler):
#     def _relabel_axis(self):
#         labels = self.ax.get_ymajorticklabels()
#         for label in labels:
#             color = self.file_group_mapping.loc[self.file_group_mapping['file_name'] ==label.get_text(), 'group_color']
#             label = label.set_color(str(color.values[0]))


#     def draw_mds(self, clusters):
#         print(f'Drawing MDS.')
#         df = MDS(n_components=2, dissimilarity='precomputed', random_state=6, metric=True).fit_transform(self.mx)
#         df = pd.DataFrame(df, columns=['comp1', 'comp2'], index=self.mx.index)
#         df = df.merge(self.file_group_mapping, how='inner', left_index=True, right_on='file_name', validate='one_to_one')
#         df = df.merge(clusters, how='inner', left_on='file_name', right_index=True, validate='1:1')

#         def _group_cluster_color(row):
#             color = None
#             if row['group_color'] == 'b' and row['cluster'] == 0:
#                 color = 'darkblue'
#             elif row['group_color'] == 'b' and row['cluster'] == 1:
#                 color = 'royalblue'
#             elif row['group_color'] == 'r' and row['cluster'] == 0:
#                 color = 'crimson'
#             #elif row['group_color'] == 'r' and row['cluster'] == 0:
#             else:
#                 color = 'deeppink'
#             return color

#         df['group_cluster_color'] = df.apply(_group_cluster_color, axis=1)


#         fig = plt.figure(figsize=(5,5))
#         ax = fig.add_subplot(1,1,1)
#         plt.scatter(df['comp1'], df['comp2'], color=df['group_cluster_color'], s=2, label="MDS")
#         plt.title = self.plot_name
#         self.save(plt, 'kmedoids-MDS', dpi=500)


