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
from .mxviz import MxViz
from .nkviz import NkViz
from .cluster_utils import CombinationInfo, MetadataHandler
from .cluster import ClusterBase
from .combinations import InfoHandler, CombinationsBase

import logging
logging.basicConfig(level=logging.DEBUG)



class ExpBase(InfoHandler):
    def __init__(self, language, cmode):
        super().__init__(language, True, cmode)
        self.add_subdir(f'{self.cmode}comb')


    def get_info_path(self, info: str):
        return os.path.join(self.subdir, f'info-{info}.pkl')
    

    def load_info(self, comb_info):
        pickle_path = self.get_info_path(comb_info)              
        with open(pickle_path, 'rb') as pickle_file:
            info = pickle.load(pickle_file)
            return info
        

    def get_top_combinations(self, exp):
        '''
        Get combinations with the best evaluation scores, load their info from file.
        '''
        topdict = TopEval(self.language, self.cmode, self.combinations_path, exp).get_topk()

        for tinfo, plttitle in topdict.items():
            comb_info, attr = tinfo.rsplit('_', 1)
            info = self.load_info(comb_info)
            
            info.add('attr', attr)
            metadf = self.merge_dfs(self.metadf, info.clusterdf)
            metadf = self.mh.add_cluster_color_and_shape(metadf)
            info.add('metadf', metadf)
            yield info, plttitle


    def run_experiments(self):
        # Default values
        if self.language == 'eng':
            maxsize = 550
        else:
            maxsize = 500
        cat_evalcol = 'ARI'
        cont_evalcol = 'logreg-accuracy' ################balanced acc, logreg_acc

        if self.cmode == 'mx':
            evalcol = 'silhouette_score'
        else:
            evalcol = 'modularity'



        topk = {'maxsize': maxsize, 'cat_evalcol': cat_evalcol, 'cont_evalcol': cont_evalcol, 'special': True}
        topcanon = {'maxsize': maxsize, 'cat_evalcol': cat_evalcol, 'cont_evalcol': cont_evalcol, 'attr': ['canon'], 'special': False}

        # Internal evaluation criterion
        interesting_attrs = ['author', 'gender', 'canon', 'year']
        intfull = {'maxsize': self.nr_texts, 'attr': interesting_attrs, 'evalcol': evalcol, 'special': True}
        intmax = {'maxsize': maxsize, 'attr': interesting_attrs, 'evalcol': evalcol, 'special': True}


        exps = {'intfull': intfull, 'intmax': intmax, 'topk': topk, 'topcanon': topcanon}
        for expname, expd in exps.items():
            self.visualize(expname, expd)


class MxExp(ExpBase):
    def __init__(self, language):
        super().__init__(language, 'mx')


    def visualize(self, expname, expd):
        # topkpath = 'mxtopk.pkl'
        # if os.path.exists(topkpath):
        #     print(topkpath)
        #     with open(topkpath, 'rb') as file:
        #         topk_comb = pickle.load(file)
        #         print('loaded topk')
        # else:
        #     topk_comb = list(self.get_top_combinations(expd))
        #     with open(topkpath, 'wb') as file:
        #         pickle.dump(topk_comb, file)
        #         print('created topk')


        for topk in self.get_top_combinations(expd):
        # for topk in topk_comb:
            info, plttitle = topk
            print(info.as_string())
            if 'special' in expd:
                info.add('special', 'canon')

            # Get matrix
            cb = CombinationsBase(self.language, add_color=False, cmode='mx')
            mx = [mx for mx in cb.mxs if mx.name == info.mxname]
            assert len(mx) == 1
            mx = mx[0]
            info.add('order', 'olo')
            viz = MxViz(self.language, mx, info, plttitle=plttitle, expname=expname)
            viz.visualize()



class NkExp(ExpBase):
    def __init__(self, language):
        super().__init__(language, 'nk')


    def visualize(self, expname, expd):
        # topkpath = 'nktopk.pkl' #############################
        # if os.path.exists(topkpath):
        #     with open(topkpath, 'rb') as file:
        #         topk_comb = pickle.load(file)
        # else:
        #     topk_comb = list(self.get_top_combinations(exp))
        #     with open(topkpath, 'wb') as file:
        #         pickle.dump(topk_comb, file)

        for topk in self.get_top_combinations(expd):
        # for topk in topk_comb:
            info, plttitle = topk
            print(info.as_string())
            if 'special' in expd:
                info.add('special', 'canon')
            network = NXNetwork(self.language, path=info.spmx_path)
            viz = NkViz(self.language, network, info, plttitle=plttitle, expname=expname)          
            viz.visualize()



class TopEval(DataHandler):
    def __init__(self, language, cmode, combinations_path, exp):
        super().__init__(language=language, output_dir='similarity', data_type='csv')
        self.cmode = cmode
        self.combinations_path = combinations_path
        self.exp = exp
        self.ntop = 1


    def extend_sizes_col(self, df):
        '''
        If a size occurs multiple times, it is written as "ntimes x size". If a size occurs only once, it is written as the size only.
        Example: One cluster of size 300 and two clusters of size 100
        'clst_sizes' columns contains string in format: "300, 2x100"
        This function converts it to "300, 100, 100"
        '''
        def compressed_to_list(x):
            sizes = []
            x = ''.join(x.split())
            x = x.split(',')
            for size in x:
                if 'x' in size:
                    n, nr = size.split('x')
                    l = [int(nr)]*int(n)
                    sizes.extend(l)
                else:
                    sizes.append(int(size))
            return sizes
        
        df['clst_sizes_ext'] = df['clst_sizes'].apply(compressed_to_list)
        return df
    

    def filter_clst_sizes(self, df):
        '''
        Find the size of the biggest cluster.
        Filter for rows where size of biggest cluster is below threshold.
        This avoids combinations where most data points are put into one cluster.
        '''
        df['biggest_clst'] = df['clst_sizes_ext'].apply(lambda x: x[0])
        df = df[df['biggest_clst'] <= self.exp['maxsize']]
        return df
    
                                
    def make_plttitle(self, df):
        '''
        Combine relevant evaluation measures into a string to display on the plots.
        '''
        if self.cmode == 'mx':
            if self.scale == 'cat':
                cols = ['ARI', 'nmi', 'fmi', 'mean_purity', 'silhouette_score', 'nclust', 'clst_sizes']
            else:
                cols = ['silhouette_score', 'nclust', 'clst_sizes'] # 'anova_pval', 'logreg_acc',  #####################################

        elif self.cmode == 'nk':
            if self.scale == 'cat':
                cols = ['ARI', 'nmi', 'fmi', 'mean_purity', 'modularity', 'nclust', 'clst_sizes']
            else:
                cols = ['modularity', 'nclust', 'clst_sizes'] # 'anova_pval', 'logreg_acc',  ##############################

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
    

    def filter_top_rows(self, df):
        '''
        Find rows with the best evaluation scores.
        '''
        def find_next_divisible(b, s):
            # Check if b is divisible by s
            if b % s == 0:
                return b
            
            # Find the next bigger number that is divisible by s
            next_divisible = (b // s + 1) * s
            return next_divisible
        

        if 'evalcol' in self.exp:
            evalcol = self.exp['evalcol']
            # Find nr of different clusterings that will be considered
            # The same clustering has the same internal evaluation value, but multiple rows in the df to to the different attrs
            # Find the nr of clusterings depending on ntop and the nr attrs that are being considered

            # Find the next multiple of the nr attrs being considered that is bigger than ntop
            nrows = find_next_divisible(self.ntop, len(self.exp['attr']))
        else:
            nrows = self.ntop
            if self.scale == 'cat':
                evalcol = self.exp['cat_evalcol']
            elif self.scale == 'cont':
                evalcol = self.exp['cont_evalcol']


        # Filter out rows that contain string values ('invalid')
        mask = pd.to_numeric(df[evalcol], errors='coerce').notnull()
        df = df.loc[mask]

        df[evalcol] = pd.to_numeric(df[evalcol], errors='raise')
        df = df.nlargest(n=nrows, columns=evalcol, keep='all')
        print('nlargest', df.shape)
        return df 
        

    def filter_attr(self, df):
        df = df[df['attr'].isin(self.exp['attr'])]
        return df
    
    
    def prepare_dfs(self):
        dfs = []
        for scale in ['cat', 'cont']:
            self.scale = scale

            evaldir = os.path.join(self.output_dir, f'{self.cmode}eval')
            df = pd.read_csv(os.path.join(evaldir, f'{self.scale}_results.csv'), header=0)

            df = self.drop_duplicated_rows(df)
            df = self.extend_sizes_col(df)
            if 'maxsize' in self.exp:
                df = self.filter_clst_sizes(df)
            if 'attr' in self.exp:
                df = self.filter_attr(df)
            df = self.filter_top_rows(df)
            df = self.make_plttitle(df)
            dfs.append(df)
        return dfs
    

    def get_topk(self):
        '''
        Find combinations with the best evaluation scores for both categorical and continuous attributes.
        '''
        topdict = {}
        dfs = self.prepare_dfs()

        if 'evalcol' not in self.exp:
            # For external evaluation, ntop rows are kept for each cat and cont
            # Different evaluation metrics are used and they are not compareable
            for df in dfs:
                d = dict(zip(df['file_info'], df['plttitle'])) 
                topdict.update(d)
        else:
            dfs = [df[['file_info', 'plttitle', self.exp['evalcol']]] for df in dfs]
            df = pd.concat(dfs)
            # For inteval, only ntop rows are kept, because cat and cont can be compared with the same inteval measure
            df = self.filter_top_rows(df)
        return topdict