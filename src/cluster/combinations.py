
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
    def __init__(self, language, add_color=False):
        super().__init__(language=language, output_dir='similarity', data_type='csv')
        self.test = False
        self.mxs = self.load_mxs()
        
        self.mh = MetadataHandler(self.language)
        self.metadf = self.mh.get_metadata(add_color=add_color)
        self.colnames = [col for col in self.metadf.columns if not col.endswith('_color')]
        # self.colnames = ['gender', 'author', 'canon', 'year']


        # Set params for testing
        if self.test:
            self.mxs = self.mxs[3:6]
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
            

    def merge_dfs(self, metadf, clusterdf):
        # Combine file names, attributes, and cluster assignments
        metadf = pd.merge(metadf, clusterdf, left_index=True, right_index=True, validate='1:1')
        return metadf
    
    
    def evaluate_all_combinations(self):
        '''
        Create all combinations and evaluate them for all attributes.
        '''
        for combination in self.create_combinations():
            start = time.time()
            self.evaluate(combination)
            print(f'{time.time()-start}s to run all evaluations.')


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


    def get_pickle_path(self, info: str):
        return os.path.join(self.subdir, f'info-{info}.pkl')


    def load_info(self, comb_info):
        pickle_path = self.get_pickle_path(comb_info)              
        with open(pickle_path, 'rb') as pickle_file:
            info = pickle.load(pickle_file)
            return info
        
    def save_info(self, info):
        info.drop('metadf') # Only save cluster assignments and not full metadata to save space on disk
        pickle_path = self.get_pickle_path(info.as_string())
        with open(pickle_path, 'wb') as f:
            pickle.dump(info, f)
            print(info.as_string())


    def load_infos(self, comb_info):
        file_path = os.path.join(self.subdir, f'combination-{comb_info}.pkl')               
        with open(file_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            return data
        

    def get_top_combinations(self):
        '''
        Get combinations with the best evaluation scores, load their info from file.
        '''
        topdict = TopEval(self.language, self.cmode).get_topk()

        for tinfo, plttitle in topdict.items():
            comb_info, attr = tinfo.rsplit('_', 1)
            info = self.load_info(comb_info)
            
            info.add('attr', attr)
            metadf = self.merge_dfs(self.metadf, info.clusterdf)
            metadf = self.mh.add_cluster_color_and_shape(metadf)
            info.add('metadf', metadf)
            yield info, plttitle



class TopEval(DataHandler):
    def __init__(self, language, cmode):
        super().__init__(language=language, output_dir='similarity', data_type='csv')
        self.cmode = cmode
        self.top_n_rows = 3


    def filter_clst_sizes(self, df):
        '''
        Find the size of the biggest cluster.
        Filter for rows where size of biggest cluster is below threshold.
        This avoids combinations where most data points where put into the same cluster.
        '''

        def extract_first_nr(x):
            x = x.split(',')[0]
            if 'x' in x:
                n = x.split('x')[1]
            else:
                n = x
            return int(n)
        

        if self.language == 'eng':
            threshold = 550
        else:
            threshold = 500

        df['biggest_clst'] = df['clst_sizes'].apply(extract_first_nr)
        print(df[['biggest_clst', 'clst_sizes']])
        df = df[df['biggest_clst'] <= threshold]

        # mask = ~df['attr'].str.contains('author|gender', case=False, na=False) ############################3

        # Apply the mask to select the rows
        # df = df[mask]
        return df
    
                                
    def make_plttitle(self, df):
        '''
        Combine relevant evaluation measures into a string to display on the plots.
        '''
        if self.cmode == 'mx':
            if self.scale == 'cat':
                cols = ['ARI', 'nmi', 'fmi', 'mean_purity', 'silhouette_score', 'nclust', 'clst_sizes']
            else:
                cols = ['anova-pval', 'logreg-accuracy', 'silhouette_score', 'nclust', 'clst_sizes'] ##################################3

        elif self.cmode == 'nk':
            if self.scale == 'cat':
                cols = ['ARI', 'nmi', 'fmi', 'mean_purity', 'modularity', 'nclust', 'clst_sizes']
            else:
                cols = ['anova-pval', 'logreg-accuracy', 'modularity', 'nclust', 'clst_sizes']

        # Create the plttitle column
        df['plttitle'] = df[cols].apply(lambda row: ', '.join(f"{col}: {row[col]}" for col in cols), axis=1)
        return df
    

    def drop_duplicated_rows(self, df):
        '''
        Identify duplicated rows based on the "file_info" column.
        Duplicated rows can occur when evaluation is cancelled and restarted.
        Combinations that were evaluated but not saved to picked in the first call are reevaluated.
        '''
        duplicated_rows = df[df.duplicated(subset=['file_info'], keep=False)]
        print("Duplicated Rows:")
        print(duplicated_rows)

        # Keep only the first occurrence of each duplicated content in "file_info"
        df = df.drop_duplicates(subset=['file_info'], keep='first')
        return df
    

    def get_topk_rows(self, df):
        '''
        Find rows with the best evaluation scores.
        '''
        if self.scale == 'cat':
            evalcol = 'ARI'
        else:
            evalcol='logreg-accuracy'

        # Filter out rows that contain string values ('invalid')
        mask = pd.to_numeric(df[evalcol], errors='coerce').notnull()
        print(df.shape)
        df = df.loc[mask]
        print(df.shape)

        df[evalcol] = pd.to_numeric(df[evalcol], errors='raise')
        df = df.nlargest(n=self.top_n_rows, columns=evalcol, keep='all')
        return df 
        

    def get_topk(self):
        '''
        Find combinations with the best evaluation scores for both categorical and continuous attributes.
        '''
        topdict = {}
        for scale in ['cat', 'cont']:
            self.scale = scale

            evaldir = os.path.join(self.output_dir, f'{self.cmode}eval')
            df = pd.read_csv(os.path.join(evaldir, f'{self.scale}_results.csv'), header=0)

            df = self.drop_duplicated_rows(df)
            df = self.filter_clst_sizes(df)
            df = self.get_topk_rows(df)
            df = self.make_plttitle(df)

            d = dict(zip(df['file_info'], df['plttitle'])) 
            topdict.update(d)
        return topdict



class MxCombinations(CombinationsBase):
    def __init__(self, language, add_color):
        super().__init__(language, add_color)
        self.add_subdir('mxcomb') # Directory for storing intermediate results
        self.cmode = 'mx'


    # def viz_attrs(self):
    #     # Visualize attributes
    #     for mx in self.mxs:
    #         for attr in self.colnames:
    #             counter = 0
    #             for order in MxReorder.ORDERS:
    #                 info = CombinationInfo(metadf=deepcopy(self.metadf), mxname=mx.name, attr=attr, order=order)
    #                 print(info.as_string(), self.language)
    #                 # Avoid repetitions from multiple values for order for continuous attributes
    #                 if attr not in ['gender', 'author']:
    #                     if counter >= 1: 
    #                         break
    #                     else:
    #                         info.order = 'continuous'
    #                 viz = MxViz(self.language, mx, info)
    #                 viz.visualize(pltname='attrviz', plttitle='Attributes')
    #                 counter += 1


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


    def viz_topk(self):
        for topk in self.get_top_combinations():
            info, plttitle = topk
            # for order in MxReorder.ORDERS:
            #     info.add('order', order)
            #     info.add('attr', attr)
            #     viz = MxViz(self.language, mx, info)
            #     viz.visualize(pltname='clstviz', plttitle=plttitle)


            # Order parameter is not necessary for evalviz
            #info.drop('order')

            mx = [mx for mx in self.mxs if mx.name == info.mxname]
            assert len(mx) == 1
            mx = mx[0]
            info.add('order', 'olo') ########################
            viz = MxViz(self.language, mx, info, plttitle=plttitle)
            viz.visualize()



class NkCombinations(CombinationsBase):
    def __init__(self, language, add_color):
        super().__init__(language, add_color)
        self.add_subdir('nkcomb') # Directory for storing intermediate results
        self.cmode = 'nk'


    # def viz_attrs(self):
    #     for mx, sparsmode in itertools.product(self.mxs, Sparsifier.MODES.keys()):
    #         sparsifier = Sparsifier(self.language, mx, sparsmode)
    #         spars_param = Sparsifier.MODES[sparsmode]
    #         for spars_param in spars_param:
    #             mx = sparsifier.sparsify(spars_param)
    #             network = NXNetwork(self.language, mx=mx)
     
    #             for attr in self.colnames:
    #                 info = CombinationInfo(metadf = deepcopy(self.metadf), mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, attr=attr)
    #                 viz = NkViz(self.language, network, info)
    #                 viz.visualize(pltname='attrviz', plttitle='Attributes')


    def create_combinations(self):
        for mx, sparsmode in itertools.product(self.mxs, Sparsifier.MODES.keys()):
            print(mx.name)
            # substring_list = ['manhattan', 'full', 'both']
            # if any(substring in mx.name for substring in substring_list): #########################

            sparsifier = Sparsifier(self.language, mx, sparsmode)
            spars_params = Sparsifier.MODES[sparsmode]
            
            for spars_param in spars_params:
                mx, filtered_nr_edges, spmx_path = sparsifier.sparsify(spars_param)
                if filtered_nr_edges != 0: ######################### write log

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


    def viz_topk_nk(self):
        for topk in self.get_top_combinations():
            info, plttitle = topk
            network = NXNetwork(self.language, path=info.spmx_path)
            viz = NkViz(self.language, network, info, plttitle=plttitle)          
            viz.visualize(plttitle=plttitle)