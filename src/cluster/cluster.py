from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.manifold import MDS
import pandas as pd
import itertools
import numpy as np
import multiprocessing
import networkx as nx
from copy import deepcopy
from itertools import product
import concurrent.futures
from networkx import edge_betweenness_centrality as betweenness
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import asyn_lpa_communities, louvain_communities
from collections import Counter
import time
import os
import pickle
import random
random.seed(9)
import sys
sys.path.append("..")
from utils import TextsByAuthor, DataHandler
from .cluster_utils import MetadataHandler

import logging
logging.basicConfig(level=logging.INFO)


class Clusters():
    '''
    sklearn: Result of clustering algorithm is a list with cluster assingments as ints.
    networkx: Result is a list of sets, each set representing a cluster
    '''
    def __init__(self, language, cluster_alg, mx, clusters):
        self.language = language
        self.cluster_alg = cluster_alg
        self.mx = mx
        self.initial_clusts = clusters
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.preprocess()
        self.df = self.as_df()


    def preprocess(self):
        # Preprocess initial clusters
        if isinstance(self.initial_clusts, np.ndarray):
            self.initial_clusts = self.initial_clusts.tolist()


    def as_df(self):
        # sklearn returns list of ints
        if all(isinstance(cluster, int) for cluster in self.initial_clusts):
            # Zip file names and cluster assingments
            assert (self.mx.mx.columns == self.mx.mx.index).all()
            clusters = dict(zip(self.mx.mx.index, self.initial_clusts))
            clusters = pd.DataFrame(list(clusters.items()), columns=['file_name', 'cluster'])

            cluster_counts = clusters['cluster'].value_counts()
            label_mapping = {label: rank for rank, (label, count) in enumerate(cluster_counts.sort_values(ascending=False).items())}
            clusters['cluster'] = clusters['cluster'].map(label_mapping)
            
        # nx returns list of sets
        elif all(isinstance(cluster, set) for cluster in self.initial_clusts):
            # Sort the sets by length in descending order
            sorted_sets = sorted(self.initial_clusts, key=len, reverse=True)

            data = {'file_name': [], 'cluster': []}
            for i, cluster_set in enumerate(sorted_sets):
                data['file_name'].extend(cluster_set)
                data['cluster'].extend([i] * len(cluster_set))

            clusters = pd.DataFrame(data)

        else:
            raise ValueError('The format of the clusters is neither a list of integers nor a list of sets.')
        
        clusters = clusters.set_index('file_name')
        clusters = clusters.sort_index()
        return clusters
    


class ClusterBase(DataHandler):
    ALGS = None

    def __init__(self, language=None, cmode=None, cluster_alg=None):
        super().__init__(language=language, output_dir='similarity', data_type='pkl')
        self.cmode = cmode
        self.cluster_alg = cluster_alg
        self.n_jobs = 1
        self.timeout = 10 # seconds
        self.get_logfile_path()


    def get_logfile_path(self):
        self.logfile_path = self.get_file_path(file_name=f'{self.cmode}_log_clst.txt')


    def log_clst(self, info, source, outcome):
        outcomes = ['timeout', 'single', 'iso', 'success', 'noisy']
        assert outcome in outcomes

        if not os.path.exists(self.logfile_path):
            with open(self.logfile_path, 'w') as f:
                f.write('info,source,outcome\n')

        with open(self.logfile_path, 'a') as f:
            f.write(f'{info.as_string()},{source},{outcome}\n')


    def get_param_combinations(self):
        # Get params for current cluster alg
        params = self.ALGS[self.cluster_alg]

        # Check if params is empty dict
        if not bool(params):
            return [{}]
        
        else:
            # Create a list of dicts with format param_name: param_value.
            param_combs = []
            combinations_product = product(*params.values())
            # Iterate through all combinations and pass them to the function
            for combination in combinations_product:
                param_combs.append(dict(zip(params.keys(), combination)))

            return param_combs


    def get_nr_authors(self):
        return len(TextsByAuthor(self.language).nr_texts_per_author)
    

    # def run_cluster_alg(self, method, param_comb):
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         # Submit the function for execution with keyword arguments
    #         future = executor.submit(method, **param_comb)

    #         try:
    #             # Set a timeout of 3 seconds
    #             clusters = future.result(timeout=3)
    #             print("Function completed successfully:")
    #         except concurrent.futures.TimeoutError:
    #             print("Function took too long to execute and timed out.")
    #             future.cancel()
    #             print("Task was canceled before completion.")
    #             print('executors')
    #             clusters = None
    #             print(clusters)

    #     print('returning clusters')
    #     return clusters
    

    def run_cluster_alg(self, method, param_comb):
        pool = multiprocessing.Pool(processes=1)

        try:
            # Apply the function with a timeout
            result = pool.apply_async(method, kwds=param_comb)
            clusters = result.get(timeout=self.timeout)
        except multiprocessing.TimeoutError:
            # Cancel the process if it exceeds the timeout
            pool.terminate()
            self.logger.debug(f'Method has not returned within {self.timeout} seconds and has been canceled.')
            clusters = None
        finally:
            # Close the pool
            pool.close()
            pool.join()
            return clusters


    def evaluate_clusters(self, df, info, source):
        # True if clustering has been successful
        valid = True

        if self.cluster_alg == 'dbscan' and (df['cluster'] == -1).all():
            valid = False
            self.log_clst(info, source, 'noisy')
            self.logger.info(f'DBSCAN only returns noisy samples with value -1.')

        if df['cluster'].nunique() == 1:
            valid = False
            self.log_clst(info, source, 'single')
            self.logger.info(f'All data points put into the same cluster.')

        elif df['cluster'].nunique() == self.nr_texts:
            valid = False
            self.log_clst(info, source, 'iso')
            self.logger.info(f'All data points put into isolated clusters.')

        else:
            # self.log_clst(info, source, 'success')
            pass
        return valid


    def cluster(self, info, param_comb):
        self.logger.debug(f'Running clustering alg {self.cluster_alg} {", ".join([f"{key}: {value}" for key, value in param_comb.items()])}.')

        # Get method name, which is the same as the name of the clustering algorithm
        method = getattr(self, self.cluster_alg)
        clusters = self.run_cluster_alg(method, param_comb)

        if clusters is None:
            self.log_clst(info, 'clst', 'timeout')
            valid = False
        else:
            clusters = Clusters(self.language, self.cluster_alg, self.mx, clusters)
            valid = self.evaluate_clusters(clusters.df, info, 'clst')

        if valid:
            return clusters
        else:
            return None

        
        

class MxCluster(ClusterBase):
    '''
    Clustering on a similarity matrix
    '''
    ALGS = {
        'hierarchical': {
            'nclust': [5, 10, 20, 50],
            'method': ['single', 'weighted', 'centroid','ward', 'median', 'average', 'complete'],
            },
        'spectral': {
            'nclust': [5, 10, 20, 50],
            },
        'kmedoids': {
            'nclust': [5, 10, 20, 50],
            },
        'dbscan': {
            'eps': [0.1, 0.3, 0.5, 0.7, 0.9],
            'minsamples': [5, 10, 30],
            },
    }


    def __init__(self, language=None, cluster_alg=None, mx=None):
        super().__init__(language=language, cmode='mx', cluster_alg=cluster_alg)
        self.mx = mx

    

    def spectral(self, **kwargs):
        clusters = SpectralClustering(n_clusters=kwargs['nclust'], eigen_solver='arpack', random_state=11, affinity='precomputed', assign_labels='kmeans', n_jobs=self.n_jobs).fit_predict(self.mx.mx)
        return clusters


    def hierarchical(self, **kwargs):
        # Distance matrix needed as input
        # Use distance matrix for linkage function
        mx = deepcopy(self.mx)
        mx.dist_to_condensed()

        linkage_matrix = linkage(mx.dmx, method=kwargs['method'])

        # Create flat clusters form hierarchical tree
        flat_clusters = fcluster(linkage_matrix, t=kwargs['nclust'], criterion='maxclust')
        return flat_clusters.tolist()


    def kmedoids(self, **kwargs):
        # According to kmedoids documentation, a 'kernel matrix' is required as input when metric is 'precomputed'. 
        # 'Kernel matrix' is not defined.
        # It is the same as a distance matrix (see experiment kmedoids-input-check.py)
        
        kmedoids = KMedoids(n_clusters=kwargs['nclust'], metric='precomputed', method='pam', init='build', random_state=8)
        clusters = kmedoids.fit_predict(self.mx.dmx)
        return clusters 
    
    def dbscan(self, **kwargs):
        d = DBSCAN(eps=kwargs['eps'], min_samples=kwargs['minsamples'], metric='precomputed', n_jobs=self.n_jobs)
        clusters = d.fit_predict(self.mx.dmx)
        return clusters



class NkCluster(ClusterBase):
    ALGS = {
        # 'girvan': { # Too slow
        #     'noparam': [None], # Girvan, Newman (2002): Community structure in social and biological networks
        #     },            
        'alpa': {},
        'louvain': {
            'resolution': [100, 10, 1, 0.1, 0.01],
            },
    }


    def __init__(self, language, cluster_alg=None, network=None):
        super().__init__(language=language, cmode='nk', cluster_alg=cluster_alg)
        self.network = network
        # Network can be None if class is only created to get parameter combination
        if self.network is not None:
            self.mx = network.mx
            self.graph = network.graph


    def alpa(self, **kwargs):
        return list(asyn_lpa_communities(self.graph))

    def louvain(self, **kwargs):
        return louvain_communities(self.graph, weight='weight', seed=11, resolution=kwargs['resolution'])
    

    def girvan(self, **kwargs):
        # girvan_newman

        # def most_central_edge(graph):
        #     # Utilize edge weights when choosing an edge with the highest betweenness centrality
        #     centrality = betweenness(graph, weight="weight")
        #     return max(centrality, key=centrality.get)
        
        # comp = girvan_newman(self.graph, most_valuable_edge=most_central_edge)
        start = time.time()
        comp = girvan_newman(self.graph)

        # limited = itertools.takewhile(lambda clusters: len(clusters) == self.attr_params[self.attr], comp)
        # Problem: minimum number of clusters bigger than predefined cluster sizes
        for communities in itertools.islice(comp, 1):
            clusters = tuple(sorted(clusters) for clusters in communities)
        print(f'{time.time()-start}s to calculate gn alg.')
        return clusters
