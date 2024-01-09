
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
from .mxviz import MxReorder, MxViz
from .nkviz import NkViz
from .sparsifier import Sparsifier
from .cluster_utils import CombinationInfo, MetadataHandler

import logging
logging.basicConfig(level=logging.DEBUG)


class CombinationsBase(DataHandler):
    def __init__(self, language):
        super().__init__(language=language, output_dir='similarity', data_type='csv')
        self.test = False ####################
        self.top_n_rows = 5 # How many combinations are considered top results ###############

        self.mxs = self.load_mxs()
        
        mh = MetadataHandler(self.language)
        self.metadf = mh.get_metadata()

        self.colnames = [col for col in self.metadf.columns if not col.endswith('_color')]
        self.colnames = ['gender', 'author', 'canon', 'year']


        # Set params for testing
        if self.test:
            self.mxs = self.mxs[3:6] ################
            self.colnames = ['gender', 'canon']
            MxReorder.ORDERS = ['olo']
            MxCluster.ALGS = {
                'hierarchical': {
                    'nclust': [2],
                    'method': ['single'],
                },
                # 'spectral': {
                #     'nclust': [2],
                # },
                # 'kmedoids': {
                #     'nclust': [2],
                # },
                # 'dbscan': {
                #     'eps': [0.01],
                #     'min_samples': [5],
                # },
            }
            NkCluster.ALGS = {
                'louvain': {
                    'resolution': [1],
                    },
            }
            Sparsifier.MODES = {
                #'authormax': [None],
                'threshold': [0.9],
                #'simmel': [(50, 100)],
            }


    # def load_processed(self, cmode):
    #     self.proc_path = self.get_file_path(file_name=f'log-processed-{cmode}.txt')
    #     # Load all combination infos that have already been run from file
    #     if os.path.exists(self.proc_path):
    #         f = open(self.proc_path, 'r')
    #         infos = [line.strip() for line in f.readlines()]
    #     else:
    #         infos = []
    #     return infos


    # def write_processed(self, info):
    #     # Write combination info that was just run to file
    #     with open(self.proc_path, 'a') as f:
    #         if isinstance(info, str):
    #             f.write(f'{info}\n')
    #         else:
    #             f.write(f'{info.as_string()}\n')


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


    def filter_biggest_clust(self, df):
        # Filter rows where the biggest cluster is below a threshold
        # threshold = round(0.8 * self.nr_texts) # round to int ##################
        if self.language == 'eng':
            threshold = 550
        else:
            threshold = 500
        df['biggest_clust'] = df['clst_str'].apply(lambda x: x.split(',')[0] if 'label' in x else None)
        df['biggest_clust'] = df['biggest_clust'].apply(lambda x: int(x.split('-')[1]))
        df = df[df['biggest_clust'] >= threshold]
        return df
    
                                
    def get_topk(self, cmode):
        evaldir = os.path.join(self.output_dir, f'{cmode}eval')

        cat = pd.read_csv(os.path.join(evaldir, 'cat_results.csv'), header=0)
        cont = pd.read_csv(os.path.join(evaldir, 'cont_results.csv'), header=0)

        cat = self.filter_biggest_clust(cat)
        cont = self.filter_biggest_clust(cont)

        cat = cat.nlargest(n=self.top_n_rows, columns='ARI')
        cont = cont.nlargest(n=self.top_n_rows, columns='logreg-accuracy')
        return cat, cont
    

    def load_combinations(self, comb_info):
        file_path = os.path.join(self.subdir, f'combination-{comb_info}.pkl')               
        with open(file_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file) # [network, clusters, info]
            return data


class MxCombinations(CombinationsBase):
    def __init__(self, language):
        super().__init__(language)
        self.add_subdir('mxcomb') # Directory for storing intermediate results
        self.cmode = 'mx'


    def viz_attrs(self):
        # Visualize attributes
        for mx in self.mxs:
            for attr in self.colnames + ['noattr']:
                counter = 0
                for order in MxReorder.ORDERS:
                    info = CombinationInfo(metadf=deepcopy(self.metadf), mxname=mx.name, attr=attr, order=order)
                    print(info.as_string(), self.language)
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


    def create_combinations(self):
        for mx, cluster_alg in itertools.product(self.mxs, MxCluster.ALGS.keys()):

            sc = MxCluster(self.language, cluster_alg, mx)
            param_combs = sc.get_param_combinations()

            for param_comb in param_combs:
                info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, param_comb=param_comb)
                clusters = sc.cluster(info, param_comb)

                combination = [mx, clusters, info] 

                if clusters is not None:
                    metadf = clusters.merge_clust_meta_dfs(self.metadf)
                    info.add('metadf', metadf)
                    self.evaluate(combination)
                
                # If clusters is None, then metadf is not added to info
                with open(os.path.join(self.subdir, f"combination-{info.as_string(omit=['attr'])}.pkl"), 'wb') as pickle_file:
                    pickle.dump(combination, pickle_file)


    def evaluate(self, combination):
        mx, clusters, info = combination
        inteval = MxIntEval(mx, clusters).evaluate()
        exteval = ExtEval(self.language, 'mx', info, inteval)
        
        for attr in ['cluster'] + self.colnames: # evaluate 'cluster' first
            info.add('attr', attr)
            exteval.evaluate(attr=attr, info=info)


    def viz_topk(self):

        cat, cont = self.get_topk(self.cmode)
        cat = dict(zip(cat['file_info'], cat['plttitle'])) 
        cont = dict(zip(cont['file_info'], cont['plttitle']))

        d = {**cat, **cont}

        for tinfo, plttitle in d.items():
            comb_info, attr = tinfo.rsplit('_', 1)
            mx, clusters, info = self.load_combinations(comb_info)

            for order in MxReorder.ORDERS:
                info.add('order', order)
                info.add('attr', attr)
                viz = MxViz(self.language, mx, info)
                viz.visualize(pltname='clstviz', plttitle=plttitle)


            # Order parameter is not necessary for evalviz
            info.drop('order')
            info.add('attr', attr)
            viz = MxViz(self.language, mx, info)
            viz.visualize(pltname='evalviz', plttitle=plttitle)



class NkCombinations(CombinationsBase):
    def __init__(self, language):
        super().__init__(language)
        self.add_subdir('nkcomb') # Directory for storing intermediate results
        self.cmode = 'nk'


    def viz_attrs(self):
        for mx, sparsmode in itertools.product(self.mxs, Sparsifier.MODES.keys()):
            sparsifier = Sparsifier(self.language, mx, sparsmode)
            spars_param = Sparsifier.MODES[sparsmode]
            for spars_param in spars_param:
                mx = sparsifier.sparsify(spars_param)
                network = NXNetwork(self.language, mx=mx)
     
                for attr in self.colnames:
                    info = CombinationInfo(metadf = deepcopy(self.metadf), mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, attr=attr)
                    viz = NkViz(self.language, network, info)
                    viz.visualize(pltname='attrviz', plttitle='Attributes')


    def create_combinations(self):
        for mx, sparsmode in itertools.product(self.mxs, Sparsifier.MODES.keys()):

            sparsifier = Sparsifier(self.language, mx, sparsmode)
            spars_params = Sparsifier.MODES[sparsmode]
            
            for spars_param in spars_params:
                mx, filtered_nr_edges = sparsifier.sparsify(spars_param)
                if filtered_nr_edges != 0: ######################### write log

                    for cluster_alg in  NkCluster.ALGS.keys():
                        network = NXNetwork(self.language, mx=mx)
                        nc = NkCluster(self.language, cluster_alg, network)
                        param_combs = nc.get_param_combinations()
                        
                        for param_comb in param_combs:
                            info = CombinationInfo(mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, cluster_alg=cluster_alg, param_comb=param_comb)
                            clusters = nc.cluster(info, param_comb)
                            combination = [network, clusters, info] 

                            if clusters is not None:
                                metadf = clusters.merge_clust_meta_dfs(self.metadf)
                                info.add('metadf', metadf)
                                self.evaluate(combination)
                            
                            # If clusters is None, then metadf is not added to info
                            with open(os.path.join(self.subdir, f"combination-{info.as_string(omit=['attr'])}.pkl"), 'wb') as pickle_file:
                                pickle.dump(combination, pickle_file)


    def evaluate(self, combination):
        network, clusters, info = combination
        inteval = NkIntEval(network, clusters, info.cluster_alg, info.param_comb).evaluate()
        exteval = ExtEval(self.language, self.cmode, info, inteval)
        
        for attr in ['cluster'] + self.colnames: # evaluate 'cluster' first
            info.add('attr', attr)
            exteval.evaluate(attr=attr, info=info)

                                
    def viz_topk(self):
        cat, cont = self.get_topk(self.cmode)
        cat = dict(zip(cat['file_info'], cat['plttitle'])) 
        cont = dict(zip(cont['file_info'], cont['plttitle']))

        d = {**cat, **cont}

        for tinfo, plttitle in d.items():
            comb_info, attr = tinfo.rsplit('_', 1)
            network, clusters, info = self.load_combinations(comb_info)

            viz = NkViz(self.language, network, info)

            info.add('attr', 'cluster')
            viz.visualize(pltname='clstviz', plttitle=plttitle)

            info.add('attr', attr)
            viz.visualize(pltname='evalviz', plttitle=plttitle)

