import matplotlib.pyplot as plt
import pandas as pd
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
from .network import NXNetwork
from .cluster import ClusterBase
from .mxviz import MxViz
from .nkviz import NkViz
from .cluster_utils import CombinationInfo, MetadataHandler
from .combinations import InfoHandler, CombinationsBase

import logging
logging.basicConfig(level=logging.DEBUG)



class ExpBase(InfoHandler):
    def __init__(self, language, cmode):
        super().__init__(language, True, cmode)
        self.add_subdir(f'{self.cmode}exp')


    def get_info_path(self, info: str):
        return os.path.join(self.subdir, f'info-{info}.pkl')
    

    def load_info(self, comb_info):
        pickle_path = self.get_info_path(comb_info)              
        with open(pickle_path, 'rb') as pickle_file:
            info = pickle.load(pickle_file)
            return info
        

    def get_top_combinations(self):
        '''
        Get combinations with the best evaluation scores, load their info from file.
        '''
        topdict = TopEval(self.language, self.cmode, self.combinations_path).get_topk()

        for tinfo, plttitle in topdict.items():
            comb_info, attr = tinfo.rsplit('_', 1)
            info = self.load_info(comb_info)
            
            info.add('attr', attr)
            metadf = self.merge_dfs(self.metadf, info.clusterdf)
            metadf = self.mh.add_cluster_color_and_shape(metadf)
            info.add('metadf', metadf)
            yield info, plttitle


class MxExp(ExpBase):
    def __init__(self, language):
        super().__init__(language, 'mx')

    def viz_topk(self):
        topkpath = 'mxtopk.pkl' #############################
        if os.path.exists(topkpath):
            print(topkpath)
            with open(topkpath, 'rb') as file:
                topk_comb = pickle.load(file)
                print('loaded topk')
        else:
            topk_comb = list(self.get_top_combinations())
            with open(topkpath, 'wb') as file:
                pickle.dump(topk_comb, file)
                print('created topk')


        # for topk in self.get_top_combinations(): ###################
        for topk in topk_comb:
            info, plttitle = topk
            print(info.as_string())

            # Get matrix
            cb = CombinationsBase(self.language, add_color=False, cmode='mx')
            mx = [mx for mx in cb.mxs if mx.name == info.mxname]
            assert len(mx) == 1
            mx = mx[0]
            info.add('order', 'olo') ########################
            viz = MxViz(self.language, mx, info, plttitle=plttitle)
            viz.visualize()


class NkExp(ExpBase):
    def __init__(self, language):
        super().__init__(language, 'nk')

    def viz_topk(self):
        topkpath = 'nktopk.pkl' #############################
        if os.path.exists(topkpath):
            with open(topkpath, 'rb') as file:
                topk_comb = pickle.load(file)
        else:
            topk_comb = list(self.get_top_combinations())
            with open(topkpath, 'wb') as file:
                pickle.dump(topk_comb, file)

        # for topk in self.get_top_combinations():
        for topk in topk_comb:
            info, plttitle = topk
            print(info.as_string())
            network = NXNetwork(self.language, path=info.spmx_path)
            viz = NkViz(self.language, network, info, plttitle=plttitle)          
            viz.visualize()







class TopEval(DataHandler):
    def __init__(self, language, cmode, combinations_path):
        super().__init__(language=language, output_dir='similarity', data_type='csv')
        self.cmode = cmode
        self.combinations_path = combinations_path
        self.top_n_rows = 2


    def check_completeness(self, cat, cont):
        with open(self.combinations_path, 'r') as file:
            nlines = sum(1 for line in file)
        mh = MetadataHandler(self.language)
        metadf = mh.get_metadata(add_color=False)
        nfeatures = metadf.shape[1]
        npossible = nfeatures * nlines

        # Combinations where clustering alg failed
        cluster_logfile = ClusterBase(self.language, self.cmode, cluster_alg=None).logfile_path
        cdf = pd.read_csv(cluster_logfile, header=0)
        cdf = cdf.loc[cdf['source'] == 'clst']
        nclst = cdf.shape[0] * nfeatures

        ncreated = cat.shape[0] + cont.shape[0]

        print(npossible == nclst + ncreated)
        print(f'npossible: {npossible}, nclst: {nclst}, ncreated: {ncreated}')


    def filter_clst_sizes(self, df):
        '''
        Find the size of the biggest cluster.
        Filter for rows where size of biggest cluster is below threshold.
        This avoids combinations where most data points are put into one cluster.
        '''
        def extract_first_nr(x):
            x = x.split(',')[0]
            if 'x' in x:
                n = x.split('x')[1]
            else:
                n = x
            return int(n)
        
        # Determine threshold based on corpus size
        if self.language == 'eng':
            threshold = 550
        else:
            threshold = 500

        df['biggest_clst'] = df['clst_sizes'].apply(extract_first_nr)
        df = df[df['biggest_clst'] <= threshold]
        return df
    
                                
    def make_plttitle(self, df):
        '''
        Combine relevant evaluation measures into a string to display on the plots.
        '''
        if self.cmode == 'mx':
            if self.scale == 'cat':
                cols = ['ARI', 'nmi', 'fmi', 'mean_purity', 'silhouette_score', 'nclust', 'clst_sizes']
            else:
                cols = ['anova-pval', 'logreg-accuracy', 'silhouette_score', 'nclust', 'clst_sizes']

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
        df = df.loc[mask]

        df[evalcol] = pd.to_numeric(df[evalcol], errors='raise')
        df = df.nlargest(n=self.top_n_rows, columns=evalcol, keep='all')
        return df 
        

    def get_topk(self):
        '''
        Find combinations with the best evaluation scores for both categorical and continuous attributes.
        '''
        topdict = {}
        dfs = []
        for scale in ['cat', 'cont']:
            self.scale = scale

            evaldir = os.path.join(self.output_dir, f'{self.cmode}eval')
            df = pd.read_csv(os.path.join(evaldir, f'{self.scale}_results.csv'), header=0)
            dfs.append(df)

            df = self.drop_duplicated_rows(df)
            df = self.filter_clst_sizes(df)
            df = self.get_topk_rows(df)
            df = self.make_plttitle(df)

            d = dict(zip(df['file_info'], df['plttitle'])) 
            topdict.update(d)
        self.check_completeness(*dfs)
        return topdict