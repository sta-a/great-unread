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
from copy import deepcopy
import itertools
import time

from utils import DataHandler
from helpers import remove_directories, delete_png_files
from cluster.create import D2vDist, Delta
from cluster.network import NXNetwork
from cluster.cluster import SimmxCluster, NetworkCluster
from cluster.evaluate import ExtEval, MxIntEval, NkIntEval
from cluster.mxviz import MxReorder, MxViz
from cluster.nkviz import NkViz
from cluster.sparsifier import Sparsifier
from cluster.cluster_utils import CombinationInfo, MetadataHandler

import logging
logging.basicConfig(level=logging.DEBUG)


class SimilarityClustering(DataHandler):
    def __init__(self, language):
        super().__init__(language=language, output_dir='similarity', data_type='csv')
        self.loggerfile = 'cluster-logging.txt'
        self.processed = self.load_processed_infos()

        self.test = False
        self.mxs = self.load_mxs()
        mh = MetadataHandler(self.language)
        self.metadf = mh.get_metadata() ############################ save to file
        self.colnames = [col for col in self.metadf.columns if not col.endswith('_color')]
        self.colnames = ['gender', 'author', 'canon', 'year']


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
                'louvain': {
                    'resolution': [1],
                    },
            }
            Sparsifier.MODES = {
                'authormax': [None], ################3
                # 'threshold': [0.9],
            # 'simmel': [(5, 10)],
            }
            NkViz.PROGS = ['sfdp']


    def load_processed_infos(self):
        # Load all combination infos that have already been run from file
        if os.path.exists('cluster-logging.txt'):
            f = open(self.loggerfile, 'r')
            infos = [line.strip() for line in f.readlines()]
        else:
            infos = []
        return infos
    
    def write_logger(self, info):
        # Write combination info that was just run to file
        with open(self.loggerfile, 'a') as f:
            f.write(f'{info.as_string()}\n')


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
            for attr in self.colnames + ['noattr']:
                counter = 0
                for order in MxReorder.ORDERS:
                    info = CombinationInfo(metadf = deepcopy(self.metadf), mxname=mx.name, attr=attr, order=order)

                    # Avoid repetitions from multiple values for order for continuous attributes
                    if attr not in ['gender', 'author']:
                        if counter >=1: 
                            break
                        elif attr == 'noattr':
                            info.order = 'noattr'
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
                info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, param_comb=param_comb)
                clusters = sc.cluster(info, **param_comb)
                inteval = MxIntEval(mx, clusters).evaluate()
                
                # Cluster visualization
                for order in MxReorder.ORDERS:
                    info = CombinationInfo(metadf = deepcopy(self.metadf), mxname=mx.name, cluster_alg=cluster_alg, attr='cluster', order=order, param_comb=param_comb)
                    print(info.as_string())
                    viz = MxViz(self.language, mx, info)
                    ee = ExtEval(self.language, 'mx', viz, clusters, info, inteval)

                # Evaluate how well clustering captures attributes
                for attr in self.colnames:
                    attrinfo = ee.evaluate(attr=attr)

        print(f'{time.time()-start}s to run 1 mx.')


    
    def network_attrviz(self):
        for mx, sparsmode in itertools.product(self.mxs, Sparsifier.MODES.keys()):
            print('\n##################################\nInfo: ', mx.name, sparsmode)
            sparsifier = Sparsifier(self.language, mx, sparsmode)
            spars_param = Sparsifier.MODES[sparsmode]
            for spars_param in spars_param:
                mx = sparsifier.sparsify(spars_param)
                network = NXNetwork(self.language, mx=mx)
     
                for prog in NkViz.PROGS:
                    for attr in self.colnames:
                        info = CombinationInfo(metadf = deepcopy(self.metadf), mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, prog=prog, attr=attr)
                        print(info.as_string())
                        viz = NkViz(self.language, network, info)
                        viz.visualize(pltname='attrviz', plttitle='Attributes')


    def network_clustering(self):
        for mx, sparsmode in itertools.product(self.mxs, Sparsifier.MODES.keys()):

            if mx.name != 'burrows-500':


                print('\n##################################\nInfo: ', mx.name, sparsmode)
                sparsifier = Sparsifier(self.language, mx, sparsmode)
                spars_params = Sparsifier.MODES[sparsmode]
                
                for spars_param in spars_params:
                    mx = sparsifier.sparsify(spars_param)

                    for cluster_alg in  NetworkCluster.ALGS.keys():
                        network = NXNetwork(self.language, mx=mx)
                        nc = NetworkCluster(self.language, cluster_alg, network)
                        param_combs = nc.get_param_combinations()
                        
                        for param_comb in param_combs:
                            info = CombinationInfo(mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, cluster_alg=cluster_alg, param_comb=param_comb)
                            if not info in self.processed:
                                clusters = nc.cluster(info, **param_comb)
                                self.write_logger(info)
                                inteval = NkIntEval(network, clusters, cluster_alg, param_comb).evaluate()

                                for prog in NkViz.PROGS:
                                    if not info in self.processed:
                                        info = CombinationInfo(metadf = deepcopy(self.metadf), mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, cluster_alg=cluster_alg, prog=prog, attr='cluster', param_comb=param_comb)
                                        print(info.as_string())
                                        viz = NkViz(self.language, network, info)
                                        ee = ExtEval(self.language, 'nk', viz, clusters, info, inteval, network)

                                        for attr in self.colnames:
                                            attrinfo = ee.evaluate(attr=attr) # Return updated info after finishing evaluation #################
                                        self.write_logger(info)

                                        print('\n-----------------------------------\n')




# delete_png_files(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkviz', '/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxviz']) 
# remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkeval', '/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxeval'])
# remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkviz'])
# if os.path.exists('cluster-logging.txt'):
#     os.remove('cluster-logging.txt')



####LOUVAIN
sn = SimilarityClustering(language='eng')
# sn.simmx_attrviz()
# sn.simmx_clustering()
# sn.network_attrviz()
sn.network_clustering()



# get colors discrete creates black?

# Elbow for internal cluster evaluation

# Hierarchical clustering:
# From scipy documentation, ignored here: Methods ‘centroid’, ‘median’, and ‘ward’ are correctly defined only if Euclidean pairwise metric is used. If y is passed as precomputed pairwise distances, then it is the user’s responsibility to assure that these distances are in fact Euclidean, otherwise the produced result will be incorrect.

# # # Similarity Graphs (Luxburg2007)
# # eta-neighborhodd graph
# # # find eta
# # eta = 0.1
# set all values below eta to 0


# %%
