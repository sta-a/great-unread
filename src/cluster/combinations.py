
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
from .cluster import MxCluster, NkCluster, ClusterBase
from .evaluate import ExtEval, MxIntEval, NkIntEval
from .sparsifier import Sparsifier
from .cluster_utils import CombinationInfo, MetadataHandler

import logging
logging.basicConfig(level=logging.DEBUG)


class InfoHandler(DataHandler):
    def __init__(self, language, output_dir='similarity', add_color=True, cmode=None, by_author=False):
        super().__init__(language=language, output_dir=output_dir, data_type='csv', by_author=by_author)
        self.add_color = add_color
        self.cmode = cmode
        if self.cmode is not None:
            self.add_subdir(f'{self.cmode}comb')
        self.by_author = by_author
        self.mh = MetadataHandler(self.language, by_author=self.by_author)
        self.metadf = self.mh.get_metadata(add_color=self.add_color)


    def get_pickle_path(self, info: str):
        return os.path.join(self.subdir, f'info-{info}.pkl')
    

    def load_info(self, comb_info):
        pickle_path = self.get_pickle_path(comb_info)              
        with open(pickle_path, 'rb') as pickle_file:
            info = pickle.load(pickle_file)
        
        repdict = {'argamon_quadratic': 'argamonquadratic', 'argamon_linear': 'argamonlinear'}
        for key, value in repdict.items():
            for attr_key, attr_value in info.__dict__.items():
                if key in str(attr_value):
                    setattr(info, attr_key, str(attr_value).replace(key, value))
        
        return info
        

    def merge_dfs(self, metadf, clusterdf):
        # Combine file names, attributes, and cluster assignments
        metadf = pd.merge(metadf, clusterdf, left_index=True, right_index=True, validate='1:1')
        return metadf     
    
        
    def save_info(self, info):
        info.drop('metadf') # Save only the cluster assignments and not complete metadata to conserve disk space.
        pickle_path = self.get_pickle_path(info.as_string())
        with open(pickle_path, 'wb') as f:
            pickle.dump(info, f)



class CombinationsBase(InfoHandler):
    '''
    This class runs all combinations of distance measures, sparsification algorithms, clustering algorithms, parameters, and evaluates the result for all attributes.
    Performed for both MDS and networks.
    '''
    def __init__(self, language, output_dir='similarity', add_color=False, cmode='nk', by_author=False, eval_only=False):
        super().__init__(language, output_dir=output_dir, add_color=add_color, cmode=cmode, by_author=by_author)
        self.eval_only = eval_only
      

        self.combinations_path = os.path.join(self.output_dir, f'{self.cmode}_log_combinations.txt')
        self.smallmx_path = os.path.join(self.output_dir, f'{self.cmode}_log_smallmx.txt')
        self.save_data(data=self.metadf, filename='metadf')
        # self.colnames = [col for col in self.metadf.columns if not col.endswith('_color')]
        self.colnames = ['gender', 'canon', 'year', 'canon-ascat', 'year-ascat'] # Don't use all features from metadf, just most important ones
        if self.by_author:
            self.colnames = self.colnames + ['canon-min', 'canon-max']
        else: 
            self.colnames = self.colnames + ['author']


        if self.eval_only:
            if self.cmode == 'nk':
                raise NotImplementedError('eval_only is has not been implemented for "nk". modularity expects different format for clusters than df.')
            # else:
            #     raise NotImplementedError('eval_only is only partially implemented for "mx". ')
        if self.eval_only:
            dc = CombDataChecker(language=self.language, cmode=self.cmode, combinations_path=self.combinations_path, by_author=self.by_author, output_dir=self.output_dir)
            cdf = dc.prepare_clst_log()
            self.cluster_log_info = set(cdf['info'])

    # def load_mxs(self):
    #     # Delta distance mxs
    #     delta = Delta(self.language)
    #     # delta.create_all_data(use_kwargs_for_fn='mode')
    #     all_delta = delta.load_all_data(use_kwargs_for_fn='mode', subdir=True)

    #     # D2v distance mxs
    #     d2v = D2vDist(language=self.language)
    #     all_d2v = d2v.load_all_data(use_kwargs_for_fn='mode', file_string=d2v.file_string, subdir=True)
    #     mxs = {**all_delta, **all_d2v}

    #     mxs_list = []
    #     for name, mx in mxs.items():
    #         mx.name = name
    #         print(name)
    #         mxs_list.append(mx)
    #     return mxs_list


    def load_mxs(self):
        # Delta distance mxs
        delta = Delta(self.language, by_author=self.by_author)
        all_delta = delta.load_all_data(use_kwargs_for_fn='mode', subdir=True)

        # D2v distance mxs
        d2v = D2vDist(language=self.language, by_author=self.by_author)
        all_d2v = d2v.load_all_data(use_kwargs_for_fn='mode', file_string=d2v.file_string, subdir=True)

        mxs = {**all_delta, **all_d2v}

        for name, mx in mxs.items():
            if 'mirror' in name:
                print('"mirror" was contained in the wrong subdir')
            mx.name = name
            yield mx


    def load_single_mx(self, mxname):
        if mxname in ['both', 'full']:
            d2v = D2vDist(language=self.language, by_author=self.by_author)
            mx = d2v.load_data(use_kwargs_for_fn='mode', file_string=d2v.file_string, subdir=True, mode=mxname)
        else:
            delta = Delta(self.language, by_author=self.by_author)
            mx = delta.load_data(use_kwargs_for_fn='mode', subdir=True, mode=mxname)      
        return mx

    
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
            inteval = MxIntEval(combination, eval_only=self.eval_only).evaluate()
        elif self.cmode == 'nk':
            inteval = NkIntEval(combination).evaluate()

        exteval = ExtEval(language=self.language, cmode=self.cmode, info=info, inteval=inteval, output_dir=self.output_dir, by_author=self.by_author)
        
        for attr in ['cluster'] + self.colnames: # evaluate 'cluster' first
            info.add('attr', attr)
            exteval.evaluate(attr=attr, info=info)
        
        # Pickle only after evaluations are done
        self.save_info(pinfo)


    def check_data(self, n_features=6):
        if not os.path.exists(self.combinations_path):
            self.log_combinations()
        dc = CombDataChecker(language=self.language, cmode=self.cmode, combinations_path=self.combinations_path, by_author=self.by_author, output_dir=self.output_dir, n_features=n_features)
        dc.check()



class MxCombinations(CombinationsBase):
    def __init__(self, language, output_dir='similarity', add_color=False, by_author=False, eval_only=False):
        super().__init__(language, output_dir=output_dir, add_color=add_color, cmode='mx', by_author=by_author, eval_only=eval_only)


    def create_combinations(self):
        mxs_generator = self.load_mxs()
        for mx, cluster_alg in itertools.product(mxs_generator, MxCluster.ALGS.keys()):

            # When clustering on embeddings, isolated nodes are ignored, and matrix of connected nodes can be too small
            if mx.mx.shape[0] > 50:

                sc = MxCluster(self.language, cluster_alg, mx, output_dir=self.output_dir, by_author=self.by_author)
                param_combs = sc.get_param_combinations()

                for param_comb in param_combs:
                    info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, param_comb=param_comb)

                    if self.eval_only:
                        if info.as_string() in self.cluster_log_info: #  invalid clusterings
                            continue
                        else:
                            info = self.load_info(info.as_string())
                            metadf = self.merge_dfs(self.metadf, info.clusterdf)
                            info.add('metadf', metadf)
                            combination = [mx, info.clusterdf, info]
                            yield combination
                    else:
                        pickle_path = self.get_pickle_path(info.as_string())              
                        if os.path.exists(pickle_path):
                            continue
                        clusters = sc.cluster(info, param_comb)

                        if clusters is not None:
                            metadf = self.merge_dfs(self.metadf, clusters.df)
                            print(metadf.shape[0], mx.mx.shape[0])
                            assert metadf.shape[0] == mx.mx.shape[0]
                            info.add('metadf', metadf)
                            info.add('clusterdf', clusters.df)
                            combination = [mx, clusters, info]
                            yield combination


    def log_combinations(self):
        self.logger.info(f'Writing all combinations to file.')
        mxs_generator = self.load_mxs()

        with open(self.combinations_path, 'w') as f:
            for mx, cluster_alg in itertools.product(mxs_generator, MxCluster.ALGS.keys()):
                # print(mx.name, cluster_alg)
                if mx.mx.shape[0] > 50:
                    sc = MxCluster(self.language, cluster_alg, mx, output_dir=self.output_dir, by_author=self.by_author)
                    param_combs = sc.get_param_combinations()

                    for param_comb in param_combs:
                        info = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, param_comb=param_comb)
                        # print(info.as_string())
                        f.write(info.as_string() + '\n')
                else:
                    # These matrices are too small to run clustering
                    with open(self.smallmx_path, 'a') as mxf:
                        mxf.write(info.as_string() + '\n')



class NkCombinations(CombinationsBase):
    def __init__(self, language, add_color, by_author=False):
        super().__init__(language, add_color=add_color, cmode='nk', by_author=by_author)
        self.noedges_path = os.path.join(self.output_dir, 'nk_noedges.txt')


    def log_noedges(self, info):
        with open(self.noedges_path, 'a') as f:
            f.write(info.as_string() + '\n')


    def create_combinations(self):
        mxs_generator = self.load_mxs()
        for mx, sparsmode in itertools.product(mxs_generator, Sparsifier.MODES.keys()):
            print(mx.name)
            # substring_list = ['manhattan', 'full', 'both']
            # if any(substring in mx.name for substring in substring_list):

            sparsifier = Sparsifier(self.language, mx, sparsmode, output_dir=self.output_dir, by_author=self.by_author)
            spars_params = Sparsifier.MODES[sparsmode]
            
            for spars_param in spars_params:
                mx, filtered_nr_edges, spmx_path = sparsifier.sparsify(spars_param)
                if filtered_nr_edges == 0:
                    einfo = CombinationInfo(mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param)
                    self.log_noedges(einfo)
                else:
                    for cluster_alg in  NkCluster.ALGS.keys():
                        network = NXNetwork(self.language, mx=mx)
                        nc = NkCluster(self.language, cluster_alg, network, output_dir=self.output_dir, by_author=self.by_author)
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
                                info.add('initial_clusts', clusters.initial_clusts)
                                combination = [network, clusters, info]

                                yield combination


    def log_combinations(self):
        mxs_generator = self.load_mxs()
        with open(self.combinations_path, 'w') as f:
            for mx, sparsmode in itertools.product(mxs_generator, Sparsifier.MODES.keys()):
                sparsifier = Sparsifier(self.language, mx, sparsmode, output_dir=self.output_dir, by_author=self.by_author)
                spars_params = Sparsifier.MODES[sparsmode]
                
                for spars_param in spars_params:
                    mx, filtered_nr_edges, spmx_path = sparsifier.sparsify(spars_param)
                    if filtered_nr_edges != 0: # ignore if no edges

                        for cluster_alg in  NkCluster.ALGS.keys():
                            nc = NkCluster(language=self.language, cluster_alg=cluster_alg, network=None, output_dir=self.output_dir, by_author=self.by_author)
                            param_combs = nc.get_param_combinations()
                            
                            for param_comb in param_combs:
                                info = CombinationInfo(mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, cluster_alg=cluster_alg, param_comb=param_comb, spmx_path=spmx_path)
                                f.write(info.as_string() + '\n')



class CombDataChecker(DataHandler):
    '''
    Check if all combinations are in evaluation files.
    eng: 95 features
    ger: 92 features
    '''
    def __init__(self, language, cmode, combinations_path, by_author, n_features='all', output_dir='similarity'):
        super().__init__(language=language, output_dir=output_dir, data_type='csv', by_author=by_author)
        self.cmode = cmode
        self.combinations_path = combinations_path
        self.by_author = by_author
        self.n_features = n_features


    def check(self):
        '''
        Find combinations with the best evaluation scores for both categorical and continuous attributes.
        '''
        dfs = []
        for scale in ['cat', 'cont']:
            self.scale = scale

            evaldir = os.path.join(self.output_dir, f'{self.cmode}eval')
            df = pd.read_csv(os.path.join(evaldir, f'{self.scale}_results.csv'), header=0, na_values=['NA'])
            df = self.drop_duplicated_rows(df)
            dfs.append(df)

        self.check_completeness(*dfs)


    def drop_duplicated_rows(self, df):
        '''
        Identify duplicated rows based on the "file_info" column.
        Duplicated rows can occur when evaluation is cancelled and restarted.
        Combinations that were evaluated but not stored as pickle in the first call are reevaluated.
        '''
        duplicated_rows = df[df.duplicated(subset=['file_info'], keep=False)]
        print("Duplicated Rows:")
        print(duplicated_rows)

        # Keep only the first occurrence of each duplicated content in "file_info"
        df = df.drop_duplicates(subset=['file_info'], keep='first')
        return df


    def prepare_possible_combs(self):
        unique_lines = set()

        with open(self.combinations_path, 'r') as file:
            for line in file:
                cleaned_line = line.strip()
                unique_lines.add(cleaned_line)

        n_unique_lines = len(unique_lines)
        return n_unique_lines
    

    def prepare_clst_log(self):
        '''
        Drop rows with duplicated info.
        Duplicated log entries can happen if the program is restarted multiple times.
        '''
        cluster_logfile = ClusterBase(self.language, self.cmode, cluster_alg=None, output_dir=self.output_dir, by_author=self.by_author).logfile_path
        cdf = pd.read_csv(cluster_logfile, header=0)
        cdf = cdf.drop_duplicates(subset=['info'], keep='first')
        cdf = cdf.loc[cdf['source'] == 'clst']
        return cdf
    

    def remove_overlap_between_clst_log_and_eval_files(self, cdf, cat, cont):
        '''
        Check if any combinations endet up in cluster log files and evaluation files
        This can happend if program is restarted and clustering was once cancelled due to timeout but successful the other time.
        '''
        cdf_set = set(cdf['info'])
        combs_in_cat = set(cat['file_info'].str.rsplit('_', n=1).str[0])
        combs_in_cont = set(cont['file_info'].str.rsplit('_', n=1).str[0])

        for cset in [combs_in_cat, combs_in_cont]:
            overlap = cdf_set.intersection(cset)
            if overlap:
                print(f"There is overlap between 'info' column and unique_file_info: {len(overlap)} combinations.")
                print(overlap)

                rows_before = len(cdf)
                cdf = cdf[~cdf['info'].isin(overlap)]
                rows_after = len(cdf)
                print(rows_before, rows_after)

            else:
                print("There is no overlap between evaluation and cluster log files.")
        
        return cdf

    def check_completeness(self, cat, cont):
        # Get combinations in cluster log file
        cdf = self.prepare_clst_log()
        cdf = self.remove_overlap_between_clst_log_and_eval_files(cdf, cat, cont)

        # Get nr possible combinations
        npossible_combs = self.prepare_possible_combs()

        if self.n_features == 'all':
            mh = MetadataHandler(self.language, by_author=self.by_author)
            metadf = mh.get_metadata(add_color=False)
            nfeatures = metadf.shape[1]
            print('nfeat', nfeatures)
        else:
            nfeatures = self.n_features
        npossible = nfeatures * npossible_combs
        print(f'npossible ({npossible}): the number of logged combinations ({npossible_combs}) (excludeing matrices that were too small) times the number of features ({nfeatures}).')

        # Combinations where clustering alg failed
        nclst = cdf.shape[0] * nfeatures
        print(f'nclst ({nclst}): the number of combinations where clustering failed ({cdf.shape[0]}), times the number of features.')

        ncreated = cat.shape[0] + cont.shape[0]
        print(f'ncreated ({ncreated}): the number of combinations in the evaluation files (cat and comb combined).')

        print(f'npossible: {npossible}, nclst: {nclst}, ncreated: {ncreated}')
        print(f'nclst + ncreated, {nclst + ncreated}')
        print(f'Evaluation files for language: {self.language} mode: {self.cmode} are complete and contain all combinations (npossible == nclst + ncreated): {npossible == nclst + ncreated}')