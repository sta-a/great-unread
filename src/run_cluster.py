# %%
%load_ext autoreload
%autoreload 2
# Don't display plots
%matplotlib agg

import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
import numpy as np
import itertools
import sys
sys.path.append("..")
from utils import DataHandler
from copy import deepcopy
import itertools
from tqdm import tqdm
import time

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
    def __init__(self, language):
        super().__init__(language=language, output_dir='similarity', data_type='csv')
        self.test = True
        self.mxs = self.load_mxs()
        mh = MetadataHandler(self.language)
        self.metadf = mh.get_metadata()
        self.colnames = [col for col in self.metadf.columns if not col.endswith('_color')]

        # Set params for testing
        if self.test:
            self.colnames = ['gender', 'canon']
            MxReorder.ORDERS = ['olo']
            SimmxCluster.ALGS = {
                'hierarchical': {
                    'nclust': [2],
                    'method': ['single'],
                },
                'spectral': {
                    'nclust': [2],
                },
                'kmedoids': {
                    'nclust': [2],
                },
                'dbscan': {
                    'eps': [0.01],
                    'min_samples': [5],
                },
            }
            NetworkCluster.ALGS = {
                'alpa': {},
                'louvain': {
                    'resolution': [1],
                },
            }
            Sparsifier.MODES = {
                'threshold': [0.9],
            }
            # NkViz.PROGS = ['fdp']


    def load_mxs(self):
        if self.test is False:
            # Delta distance mxs
            delta = Delta(self.language)
            # delta.create_all_data(use_kwargs_for_fn='mode')
            all_delta = delta.load_all_data(use_kwargs_for_fn='mode', subdir=True)


            # D2v distance mxs
            d2v = D2vDist(language=self.language)
            all_d2v = d2v.load_all_data(use_kwargs_for_fn='mode', file_string=d2v.file_string, subdir=True)
            mxs = {**all_delta, **all_d2v}

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


    def simmx_attrviz(self):
        # Visualize attributes
        for mx in self.mxs:
            for attr in self.colnames:
                counter = 0
                for order in MxReorder.ORDERS:
                    info = CombinationInfo(metadf = deepcopy(self.metadf), mxname=mx.name, attr=attr, order=order)

                    # Avoid repetitions from multiple values for order for continuous attributes
                    if attr not in ['gender', 'author']:
                        if counter >=1: 
                            break
                        else:
                            info.order = 'continuous'
                    viz = MxViz(self.language, mx, info)
                    viz.visualize(pltname='attrviz', plttitle='Attributes')
                    counter += 1


    def simmx_clustering(self):
        start = time.time()
        # Visualize and evaluate clusters
        for mx, cluster_alg in itertools.product(self.mxs, SimmxCluster.ALGS.keys()):
            sc = SimmxCluster(self.language, cluster_alg, mx)
            param_combs = sc.get_param_combinations()

            for param_comb in param_combs:
                clusters = sc.cluster(**param_comb)
                inteval = MxIntEval(mx, clusters).evaluate()
                
                # Cluster visualization
                for order in MxReorder.ORDERS:
                    info = CombinationInfo(metadf = deepcopy(self.metadf), mxname=mx.name, cluster_alg=cluster_alg, attr='cluster', order=order, param_comb=param_comb)
                    print(info.as_string())
                    viz = MxViz(self.language, mx, info)
                    ee = ExtEval(self.language, 'mx', viz, clusters, info, inteval)

                # Evaluate how well clustering captures attributes
                for attr in self.colnames:
                    ee.evaluate(attr=attr)

        print(f'{time.time()-start}s to run 1 mx.')


    
    def network_attrviz(self):
        for mx, spars in itertools.product(self.mxs, Sparsifier.MODES.keys()):
            spars_params = Sparsifier.MODES[spars]
            for spars_param in spars_params:
                sparsifier = Sparsifier(self.language, mx, spars, spars_param)
                mx = sparsifier.sparsify()
                network = NXNetwork(self.language, mx=mx)
     
                for attr in self.colnames:
                    for prog in NkViz.PROGS:
                        info = CombinationInfo(metadf = deepcopy(self.metadf), mxname=mx.name, spars=spars, spars_param=spars_param, attr=attr, prog=prog)
                        print(info.as_string())
                        viz = NkViz(self.language, network, info)
                        viz.visualize(pltname='attrviz', plttitle='Attributes')


    def network_clustering(self):
        for mx, spars, cluster_alg in itertools.product(self.mxs, Sparsifier.MODES.keys(), NetworkCluster.ALGS):
            spars_params = Sparsifier.MODES[spars]
            for spars_param in spars_params:
                sparsifier = Sparsifier(self.language, mx, spars, spars_param)
                mx = sparsifier.sparsify()

                network = NXNetwork(self.language, mx=mx)
                nc = NetworkCluster(self.language, cluster_alg, network)
                param_combs = nc.get_param_combinations()
                for param_comb in param_combs:
                    clusters = nc.cluster(**param_comb)
                    inteval = NkIntEval(network, clusters, cluster_alg, param_comb).evaluate()

                    for prog in NkViz.PROGS:
                        info = CombinationInfo(metadf = deepcopy(self.metadf), mxname=mx.name, spars=spars, spars_param=spars_param, cluster_alg=cluster_alg, attr='cluster', prog=prog, param_comb=param_comb)
                        print(info.as_string())
                        viz = NkViz(self.language, network, info)
                        ee = ExtEval(self.language, 'nk', viz, clusters, info, inteval)

                        for attr in self.colnames:
                            ee.evaluate(attr=attr)




def remove_directories(directory_paths):
    for path in directory_paths:
        try:
            shutil.rmtree(path)
            print(f"Directory '{path}' removed successfully.")
        except OSError as e:
            print(f"Error removing directory '{path}': {e}")

def delete_png_files(directory):
    # Get a list of all files in the directory
    file_list = os.listdir(directory)

    # Iterate through the files and delete those with a '.png' extension
    for filename in file_list:
        if filename.endswith('.png'):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

delete_png_files('/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkviz')          
# directories_to_remove = ['/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxviz', '/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxeval']
# remove_directories(directories_to_remove)



sn = SimilarityClustering(language='eng')
# sn.simmx_attrviz()
# sn.simmx_clustering()
# sn.network_attrviz()
sn.network_clustering()


# Attr in networkviz
# Elbow for internal cluster evaluation

# Hierarchical clustering:
# From scipy documentation, ignored here: Methods ‘centroid’, ‘median’, and ‘ward’ are correctly defined only if Euclidean pairwise metric is used. If y is passed as precomputed pairwise distances, then it is the user’s responsibility to assure that these distances are in fact Euclidean, otherwise the produced result will be incorrect.

# # # Similarity Graphs (Luxburg2007)
# # eta-neighborhodd graph
# # # find eta
# # eta = 0.1
# set all values below eta to 0



# %%
