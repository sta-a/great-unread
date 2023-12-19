# %%
# %load_ext autoreload
# %autoreload 2
# Don't display plots
# %matplotlib agg

import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
import numpy as np
import itertools
import sys
from copy import deepcopy
sys.path.append("..")
from copy import deepcopy
import itertools
import shutil
import time
import argparse


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
    def __init__(self, language, draw=True):
        super().__init__(language=language, output_dir='similarity', data_type='csv')
        self.draw = draw
        self.test = False

        self.mxs = self.load_mxs()
        
        mh = MetadataHandler(self.language)
        self.metadf = mh.get_metadata()

        self.colnames = [col for col in self.metadf.columns if not col.endswith('_color')]
        self.colnames = ['gender', 'author', 'canon', 'year']


        # Set params for testing
        if self.test:
            self.mxs = [self.mxs[0]]
            self.colnames = ['gender', 'canon']
            MxReorder.ORDERS = ['olo']
            SimmxCluster.ALGS = {
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
            NetworkCluster.ALGS = {
                'louvain': {
                    'resolution': [1],
                    },
            }
            Sparsifier.MODES = {
                #'authormax': [None],
                # 'threshold': [0.9],
                'simmel': [(50, 100)],
            }
            NkViz.PROGS = ['sfdp']


    def load_processed(self, cmode):
        self.proc_path = self.get_file_path(file_name=f'log-processed-{cmode}.txt')
        # Load all combination infos that have already been run from file
        if os.path.exists(self.proc_path):
            f = open(self.proc_path, 'r')
            infos = [line.strip() for line in f.readlines()]
        else:
            infos = []
        return infos


    def write_processed(self, info):
        # Write combination info that was just run to file
        with open(self.proc_path, 'a') as f:
            if isinstance(info, str):
                f.write(f'{info}\n')
            else:
                f.write(f'{info.as_string()}\n')


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


    def simmx_attrviz(self):
        # Visualize attributes
        for mx in self.mxs:
            for attr in self.colnames + ['noattr']:
                counter = 0
                for order in MxReorder.ORDERS:
                    info = CombinationInfo(metadf = deepcopy(self.metadf), mxname=mx.name, attr=attr, order=order)
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


    def simmx_clustering(self):
        cmode = 'mx'
        processed = self.load_processed(cmode=cmode)

        for mx, cluster_alg in itertools.product(self.mxs, SimmxCluster.ALGS.keys()):
            start = time.time()
            if mx.name not in processed:

                print('\n##################################\n', self.language)
                sc = SimmxCluster(self.language, cluster_alg, mx)
                param_combs = sc.get_param_combinations()

                for param_comb in param_combs:
                    outinfo = CombinationInfo(mxname=mx.name, cluster_alg=cluster_alg, param_comb=param_comb, metadf=deepcopy(self.metadf))
                    if outinfo.as_string() in processed:
                        pass

                    else:
                        clusters = sc.cluster(outinfo, param_comb)
                        if clusters is not None:
                            inteval = MxIntEval(mx, clusters).evaluate()

                            info = deepcopy(outinfo)
                            print(info.as_string(), self.language)
                            ee = ExtEval(self.language, 'mx', clusters, info, inteval)
                            
                                        
                            eval_info = {}
                            for attr in ['cluster'] + self.colnames:
                                info.add('attr', attr)
                                pt = ee.evaluate(attr=attr, info=info)
                                eval_info[attr] = deepcopy(pt)


                            if self.draw:
                                for attr, pt in eval_info.items(): # fit fp

                                    if attr == 'cluster':
                                        pltname='clstviz'
                                        for order in MxReorder.ORDERS:
                                            info.add('order', order)
                                            viz = MxViz(self.language, mx, info)
                                            viz.visualize(pltname=pltname, plttitle=pt.as_string(sep='\n'))
                                    else:
                                        # Order parameter is not necessary for evalviz
                                        pltname='evalviz'
                                        info.drop('order')
                                        viz = MxViz(self.language, mx, info)
                                        viz.visualize(pltname=pltname, plttitle=pt.as_string(sep='\n'))

                        self.write_processed(outinfo.as_string())
                self.write_processed(mx.name)


            print(f'{time.time()-start}s to run 1 mx.', self.language)


    def network_attrviz(self):
        for mx, sparsmode in itertools.product(self.mxs, Sparsifier.MODES.keys()):
            print('\n##################################\nInfo: ', mx.name, sparsmode, self.language)
            sparsifier = Sparsifier(self.language, mx, sparsmode)
            spars_param = Sparsifier.MODES[sparsmode]
            for spars_param in spars_param:
                mx = sparsifier.sparsify(spars_param)
                network = NXNetwork(self.language, mx=mx)
     
                for attr in self.colnames:
                    for prog in NkViz.PROGS:
                        info = CombinationInfo(metadf = deepcopy(self.metadf), mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, attr=attr, prog=prog)
                        print(info.as_string(), self.language)
                        viz = NkViz(self.language, network, info)
                        viz.visualize(pltname='attrviz', plttitle='Attributes')





    def network_clustering(self):
        # change line 135 in eval, 163, 176
        cmode = 'nk'
        processed = self.load_processed(cmode=cmode)

        for mx, sparsmode in itertools.product(self.mxs, Sparsifier.MODES.keys()):
            start = time.time()

            if mx.name not in processed:
                print('\n##################################\n', self.language)
                sparsifier = Sparsifier(self.language, mx, sparsmode)
                spars_params = Sparsifier.MODES[sparsmode]
                
                for spars_param in spars_params:
                    mx, original_nr_edges, filtered_nr_edges = sparsifier.sparsify(spars_param)
                    if filtered_nr_edges != 0: ######################### write log

                        for cluster_alg in  NetworkCluster.ALGS.keys():
                            network = NXNetwork(self.language, mx=mx)
                            nc = NetworkCluster(self.language, cluster_alg, network)
                            param_combs = nc.get_param_combinations()
                            
                            for param_comb in param_combs:
                                outinfo = CombinationInfo(mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, cluster_alg=cluster_alg, param_comb=param_comb, metadf=deepcopy(self.metadf))
                                if outinfo.as_string() in processed:
                                    pass
                                else:
                                    clusters = nc.cluster(outinfo, param_comb)
                                    
                                    if clusters is not None:
                                        inteval = NkIntEval(network, clusters, cluster_alg, param_comb).evaluate()

                                        info = deepcopy(outinfo)
                                        print(info.as_string(), self.language)
                                        ee = ExtEval(self.language, 'nk', clusters, info, inteval)
                                        info = deepcopy(ee.info) # clusters added to metadf
                                        
                                        eval_info = {}
                                        for attr in ['cluster'] + self.colnames:
                                            info.add('attr', attr)
                                            pt = ee.evaluate(attr=attr, info=info)
                                            eval_info[attr] = deepcopy(pt)


                                        if self.draw:
                                            for attr, pt in eval_info.items():
                                                for prog in NkViz.PROGS:
                                                    info.add('prog', prog)
                                                    viz = NkViz(self.language, network, info)
                                                    if attr == 'cluster':
                                                        pltname='clstviz'
                                                    else:
                                                        pltname='evalviz'
                                                    viz.visualize(pltname=pltname, plttitle=pt.as_string(sep='\n'))

                                self.write_processed(outinfo.as_string())
                    self.write_processed(mx.name)
            print(f'{time.time()-start}s to run 1 mx.', self.language)


    def get_all_infos(self):
        all_infos = []

        for mx, sparsmode in itertools.product(self.mxs, Sparsifier.MODES.keys()):
            sparsifier = Sparsifier(self.language, mx, sparsmode)
            spars_params = Sparsifier.MODES[sparsmode]
            
            for spars_param in spars_params:
                for cluster_alg in  NetworkCluster.ALGS.keys():
                    nc = NetworkCluster(self.language, cluster_alg, network=None)
                    param_combs = nc.get_param_combinations()
                    
                    for param_comb in param_combs:
                        for attr in self.colnames:
                            info = CombinationInfo(mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, cluster_alg=cluster_alg, param_comb=param_comb, attr=attr)
                            all_infos.append(info)


    def filter_biggest_clust(self, df):
        # Filter rows where the biggest cluster is below a threshold
        threshold = round(0.9 * self.nr_texts) # round to int
        df['biggest_clust'] = df['clst_str'].apply(lambda x: x.split(',')[0] if 'label' in x else None)
        df['biggest_clust'] = df['biggest_clust'].apply(lambda x: int(x.split('-')[1]))
        df = df[df['biggest_clust'] >= threshold]
        return df
    
                                
    def get_topk(self, cmode):
        dirpath = f'/home/annina/scripts/great_unread_nlp/data/similarity/{self.language}/{cmode}eval'
        nrows = 30

        cat = pd.read_csv(os.path.join(dirpath, 'cat_results.csv'), header=0)
        cont = pd.read_csv(os.path.join(dirpath, 'cont_results.csv'), header=0)

        cat = self.filter_biggest_clust(cat)
        cont = self.filter_biggest_clust(cont)

        cat = cat.nlargest(n=nrows, columns='ARI')
        top_infos_cat = cat['file_info'].tolist()

        cont = cont.nlargest(n=nrows, columns='logreg-accuracy')
        top_infos_cont = cont['file_info'].tolist()
        return top_infos_cat + top_infos_cont, cat, cont
    

    def get_best_viz(self, cmode):
        # Copy topk files to dir
        top_infos, cat, cont = self.get_topk(cmode)
        inpath = f'/home/annina/scripts/great_unread_nlp/data/similarity/{self.language}/{cmode}viz'
        topk_dir = f'/home/annina/scripts/great_unread_nlp/data/similarity/{self.language}/{cmode}_topk'

        cat.to_csv(os.path.join(topk_dir, 'cat.csv'), header=True)
        cont.to_csv(os.path.join(topk_dir, 'cont.csv'), header=True)


        # Create the subdir 'topk' if it doesn't exist
        os.makedirs(topk_dir, exist_ok=True)

        # Iterate through files in the directory
        for filename in os.listdir(inpath):
            file_path = os.path.join(inpath, filename)
            
            # Check if the filename or any substring is in the list
            if any(substring in filename for substring in top_infos):
                # Copy the file to the 'topk' directory
                shutil.copy(file_path, os.path.join(topk_dir, filename))


    def helper_network_clustering(self):
        '''
        Delete!!!
        Set output dir in networkviz!!!
        '''
        # change line 135 in eval, 163, 176
        cmode = 'nk'
        top_infos, cat, cont = self.get_topk(cmode)

        for mx, sparsmode in itertools.product(self.mxs, Sparsifier.MODES.keys()):
            start = time.time()

            print('\n##################################\n', self.language)
            sparsifier = Sparsifier(self.language, mx, sparsmode)
            spars_params = Sparsifier.MODES[sparsmode]
            
            for spars_param in spars_params:
                mx, original_nr_edges, filtered_nr_edges = sparsifier.sparsify(spars_param)
                if filtered_nr_edges != 0: ######################### write log

                    for cluster_alg in  NetworkCluster.ALGS.keys():
                        network = NXNetwork(self.language, mx=mx)
                        nc = NetworkCluster(self.language, cluster_alg, network)
                        param_combs = nc.get_param_combinations()
                        
                        for param_comb in param_combs:
                            outinfo = CombinationInfo(mxname=mx.name, sparsmode=sparsmode, spars_param=spars_param, cluster_alg=cluster_alg, param_comb=param_comb, metadf=deepcopy(self.metadf))

                            clusters = nc.cluster(outinfo, param_comb)
                            
                            if clusters is not None:
                                inteval = NkIntEval(network, clusters, cluster_alg, param_comb).evaluate()

                                info = deepcopy(outinfo)
                                print(info.as_string(), self.language)
                                ee = ExtEval(self.language, 'nk', clusters, info, inteval)
                                info = deepcopy(ee.info) # clusters added to metadf
                                
                                eval_info = {}
                                for attr in ['cluster'] + self.colnames:
                                    info.add('attr', attr)
                                    if info.as_string() in top_infos:
                                        eval_info[attr] = 'top plots'


                                if eval_info: # Draw if info is in top infos
                                    for attr, pt in eval_info.items(): # Fix pt
                                        for prog in NkViz.PROGS:
                                            info.add('prog', prog)
                                            viz = NkViz(self.language, network, info)
                                            if attr == 'cluster':
                                                pltname='clstviz'
                                            else:
                                                pltname='evalviz'
                                            viz.visualize(pltname=pltname, plttitle=pt)

            print(f'{time.time()-start}s to run 1 mx.', self.language)


# remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkeval', '/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxeval'])
# delete_png_files(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkviz', '/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxviz']) 
# remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkviz'])
# remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/clusters'])
# logfiles = ['/home/annina/scripts/great_unread_nlp/data/similarity/eng/log_clst.txt', '/home/annina/scripts/great_unread_nlp/data/similarity/eng/log-processed.txt']
# for i in logfiles:
#     if os.path.exists(i):
#         os.remove(i)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--language', type=str)
    parser.add_argument('--mode', type=str)

    args = parser.parse_args()

    language = args.language
    mode = args.mode

    print(f"Selected language: {language}")
    print(f"Selected mode: {mode}")


    # sc = SimilarityClustering(language=language, draw=True)
    # if mode == 'mxc':
    #     sc.simmx_clustering()
    # elif mode == 'mxv':
    #     sc.simmx_attrviz()
    # elif mode == 'nkc':
    #     sc = SimilarityClustering(language=language, draw=False)
    #     sc.network_clustering()
    # elif mode == 'nkv':
    #     sc.network_attrviz()



    sc = SimilarityClustering(language=language, draw=True)
    if mode == 'nkbest':
        sc.helper_network_clustering()
        sc.get_best_viz('nk')
    elif mode == 'mxbest':
        sc.get_best_viz('mx')
        



# lst = ['ger']
# for language in lst:
#     sc = SimilarityClustering(language=language, draw=True)
#     sc.simmx_clustering()
#     print('----------------------\n\n')

# for language in lst:
#     sc = SimilarityClustering(language=language, draw=False)#########################
#     sc.network_clustering()
#     print('----------------------\n\n')

# for language in lst:
#     sc = SimilarityClustering(language=language, draw=True)
#     sc.simmx_attrviz()
#     print('----------------------\n\n')

# for language in lst:
#     sc = SimilarityClustering(language=language, draw=False)
#     sc.network_clustering()
#     print('----------------------\n\n')



# Elbow for internal cluster evaluation

# Hierarchical clustering:
# From scipy documentation, ignored here: Methods ‘centroid’, ‘median’, and ‘ward’ are correctly defined only if Euclidean pairwise metric is used. If y is passed as precomputed pairwise distances, then it is the user’s responsibility to assure that these distances are in fact Euclidean, otherwise the produced result will be incorrect.

# # # Similarity Graphs (Luxburg2007)
# # eta-neighborhodd graph
# # # find eta
# # eta = 0.1
# set all values below eta to 0

# %%
