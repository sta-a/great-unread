
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
import numpy as np
import pickle
import itertools
import sys
from copy import deepcopy
sys.path.append("..")
import itertools
import shutil
import time

from utils import DataHandler
from .create import D2vDist, Delta
from .network import NXNetwork
from .cluster import MxCluster, NkCluster
from .evaluate import ExtEval, MxIntEval, NkIntEval
from .mxviz import MxReorder
from .sparsifier import Sparsifier
from .cluster_utils import CombinationInfo, MetadataHandler

import logging
logging.basicConfig(level=logging.DEBUG)


class InfoHandler(DataHandler):
    def __init__(self, language, add_color, cmode):
        super().__init__(language=language, output_dir='similarity', data_type='csv')
        self.add_color = add_color
        self.cmode = cmode
        self.mh = MetadataHandler(self.language)
        self.metadf = self.mh.get_metadata(add_color=True)
        self.combinations_path = os.path.join(self.output_dir, f'{self.cmode}_log_combinations.txt')
            

    def merge_dfs(self, metadf, clusterdf):
        # Combine file names, attributes, and cluster assignments
        metadf = pd.merge(metadf, clusterdf, left_index=True, right_index=True, validate='1:1')
        return metadf     


    def get_pickle_path(self, info: str):
        return os.path.join(self.subdir, f'info-{info}.pkl')
    
        
    def save_info(self, info):
        info.drop('metadf') # Only save cluster assignments and not full metadata to save space on disk
        pickle_path = self.get_pickle_path(info.as_string())
        with open(pickle_path, 'wb') as f:
            pickle.dump(info, f)
            print(info.as_string())


class CombinationsBase(InfoHandler):
    '''
    This class runs all combinations of distance measures, sparsification algorithms, clustering algorithms, parameters, and evaluates the result for all attributes.
    Performed for both MDS and networks.
    '''
    def __init__(self, language, add_color=False, cmode='nk'):
        super().__init__(language, add_color, cmode)
        self.test = False
        self.mxs = self.load_mxs()
        
        self.add_subdir(f'{self.cmode}comb')

        self.save_data(data=self.metadf, filename='metadf')
        self.colnames = [col for col in self.metadf.columns if not col.endswith('_color')]


        if self.test:
            self.mxs = self.mxs[3:6]
            self.colnames = ['gender', 'author', 'canon', 'year']
            MxReorder.ORDERS = ['olo']
            MxCluster.ALGS = {
                'hierarchical': {
                    'nclust': [2],
                    'method': ['single'],
                },
            }
            NkCluster.ALGS = {
                'louvain': {
                    'resolution': [1],
                    },
            }
            Sparsifier.MODES = {
                'threshold': [0.9],
            }


    def load_mxs(self):
        # Delta distance mxs
        delta = Delta(self.language)
        # delta.create_all_data(use_kwargs_for_fn='mode')
        all_delta = delta.load_all_data(use_kwargs_for_fn='mode', subdir=True)

        # D2v distance mxs
        d2v = D2vDist(language=self.language)
        all_d2v = d2v.load_all_data(use_kwargs_for_fn='mode', file_string=d2v.file_string, subdir=True)
        mxs = {**all_delta, **all_d2v}

        mxs_list = []
        for name, mx in mxs.items():
            mx.name = name
            mxs_list.append(mx)
        return mxs_list
    
    
    def evaluate_all_combinations(self):
        '''
        Create all combinations and evaluate them for all attributes.
        '''
        for combination in self.create_combinations():
            self.evaluate(combination)


    def evaluate(self, combination):
        '''
        Evaluate a single combination for all attributes.
        '''
        _, _, info = combination # network/mx, clusters, info
        pinfo = deepcopy(info)

        if self.cmode == 'mx':
            inteval = MxIntEval(combination).evaluate()
        elif self.cmode == 'nk':
            inteval = NkIntEval(combination).evaluate()

        exteval = ExtEval(self.language, self.cmode, info, inteval)
        
        for attr in ['cluster'] + self.colnames: # evaluate 'cluster' first
            info.add('attr', attr)
            print(info.as_string())
            exteval.evaluate(attr=attr, info=info)
        
        # Pickle only after evaluations are done
        self.save_info(pinfo)



class MxCombinations(CombinationsBase):
    def __init__(self, language, add_color):
        super().__init__(language, add_color, cmode='mx')


    def create_combinations(self):
        for mx, cluster_alg in itertools.product(self.mxs, MxCluster.ALGS.keys()):
            print(mx.name)

            sc = MxCluster(self.language, cluster_alg, mx)
            param_combs = sc.get_param_combinations()

            for param_comb in param_combs:
                info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, param_comb=param_comb)
                if os.path.exists(self.get_pickle_path(info.as_string())):
                    continue
                clusters = sc.cluster(info, param_comb)

                if clusters is not None:
                    print(info.as_string())
                    metadf = self.merge_dfs(self.metadf, clusters.df)
                    info.add('metadf', metadf)
                    info.add('clusterdf', clusters.df)
                    combination = [mx, clusters, info] 

                    yield combination


    def log_combinations(self):
        with open(self.combinations_path, 'w') as f:
            for mx, cluster_alg in itertools.product(self.mxs, MxCluster.ALGS.keys()):
                sc = MxCluster(self.language, cluster_alg, mx)
                param_combs = sc.get_param_combinations()

                for param_comb in param_combs:
                    info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, param_comb=param_comb)
                    f.write(info.as_string() + '\n')



class NkCombinations(CombinationsBase):
    def __init__(self, language, add_color):
        super().__init__(language, add_color, cmode='nk')
        self.noedges_path = os.path.join(self.output_dir, 'nk_noedges.txt')


    def log_noedges(self, info):
        with open(self.noedges_path, 'a') as f:
            f.write(info.as_string() + '\n')


    def create_combinations(self):
        for mx, sparsmode in itertools.product(self.mxs, Sparsifier.MODES.keys()):
            print(mx.name)
            # substring_list = ['manhattan', 'full', 'both']
            # if any(substring in mx.name for substring in substring_list):

            sparsifier = Sparsifier(self.language, mx, sparsmode)
            spars_params = Sparsifier.MODES[sparsmode]
            
            for spars_param in spars_params:
                mx, filtered_nr_edges, spmx_path = sparsifier.sparsify(spars_param)
                if filtered_nr_edges == 0:
                    einfo = CombinationInfo(mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param)
                    self.log_noedges(einfo)
                else:
                    for cluster_alg in  NkCluster.ALGS.keys():
                        network = NXNetwork(self.language, mx=mx)
                        nc = NkCluster(self.language, cluster_alg, network)
                        param_combs = nc.get_param_combinations()
                        
                        for param_comb in param_combs:
                            info = CombinationInfo(mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, cluster_alg=cluster_alg, param_comb=param_comb, spmx_path=spmx_path)
                            if os.path.exists(self.get_pickle_path(info.as_string())):
                                continue
                            clusters = nc.cluster(info, param_comb)

                            if clusters is not None:
                                print(info.as_string())
                                metadf = self.merge_dfs(self.metadf, clusters.df)
                                info.add('metadf', metadf)
                                info.add('clusterdf', clusters.df)
                                combination = [network, clusters, info]

                                yield combination


    def log_combinations(self):
        with open(self.combinations_path, 'w') as f:
            for mx, sparsmode in itertools.product(self.mxs, Sparsifier.MODES.keys()):
                sparsifier = Sparsifier(self.language, mx, sparsmode)
                spars_params = Sparsifier.MODES[sparsmode]
                
                for spars_param in spars_params:
                    mx, filtered_nr_edges, spmx_path = sparsifier.sparsify(spars_param)
                    if filtered_nr_edges != 0: # ignore if no edges

                        for cluster_alg in  NkCluster.ALGS.keys():
                            nc = NkCluster(language=self.language, cluster_alg=cluster_alg, network=None)
                            param_combs = nc.get_param_combinations()
                            
                            for param_comb in param_combs:
                                info = CombinationInfo(mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, cluster_alg=cluster_alg, param_comb=param_comb, spmx_path=spmx_path)
                                f.write(info.as_string() + '\n')


