from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.manifold import MDS
import pandas as pd
import itertools
import numpy as np
import networkx as nx
from copy import deepcopy
from itertools import product
from networkx import edge_betweenness_centrality as betweenness
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import asyn_lpa_communities, louvain_communities
from collections import Counter
import time
import random
random.seed(9)
import sys
sys.path.append("..")
from utils import TextsByAuthor

import logging
logging.basicConfig(level=logging.DEBUG)
# # Suppress logger messages by 'matplotlib.ticker
# # Set the logging level to suppress debug output
# ticker_logger = logging.getLogger('matplotlib.ticker')
# ticker_logger.setLevel(logging.WARNING)

test = True

class Clusters():
    '''
    sklearn: Result of clustering algorithm is a list with cluster assingments as ints.
    networkx: Result is a list of sets, each set representing a cluster
    '''
    def __init__(self, cluster_alg, mx, clusters):
        self.cluster_alg = cluster_alg
        self.mx = mx
        self.initial_clusts = clusters
        self.preprocess()
        self.df = self.as_df()
    

    def preprocess(self):
        # Preprocess self.initial_clusts
        if isinstance(self.initial_clusts, np.ndarray):
            self.initial_clusts = self.initial_clusts.tolist()

        if self.cluster_alg == 'dbscan' and all(element == -1 for element in self.initial_clusts):
            self.logger.error(f'DBSCAN only returns noisy samples with value -1.')

        if all(isinstance(cluster, int) for cluster in self.initial_clusts):
            self.type = 'intlist'
        elif all(isinstance(cluster, set) for cluster in self.initial_clusts):
            self.type = 'setlist'
        else:
            raise ValueError('The format of the clusters is neither a list of integers nor a list of sets.')


    def as_df(self):
        # sklearn
        if self.type == 'intlist':
            # Zip file names and cluster assingments
            assert (self.mx.mx.columns == self.mx.mx.index).all()
            clusters = dict(zip(self.mx.mx.index, self.initial_clusts))
            clusters = pd.DataFrame(list(clusters.items()), columns=['file_name', 'cluster'])

            cluster_counts = clusters['cluster'].value_counts()
            label_mapping = {label: rank for rank, (label, count) in enumerate(cluster_counts.sort_values(ascending=False).items())}
            clusters['cluster'] = clusters['cluster'].map(label_mapping)


        # nx.louvain_communities
        elif self.type == 'setlist':
            # Sort the sets by length in descending order
            sorted_sets = sorted(self.initial_clusts, key=len, reverse=True)

            data = {'file_name': [], 'cluster': []}
            for i, cluster_set in enumerate(sorted_sets):
                data['file_name'].extend(cluster_set)
                data['cluster'].extend([i] * len(cluster_set))

            clusters = pd.DataFrame(data)

        clusters = clusters.set_index('file_name')
        clusters = clusters.sort_index()
        return clusters
    
    
class ClusterBase():
    ALGS = None

    def __init__(self, language=None, cluster_alg=None):
        self.language = language
        self.cluster_alg = cluster_alg
        self.attr_params = {'gender': 2, 'author': self.get_nr_authors()}
        self.n_jobs = -1
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)


    def get_param_combinations(self):
        # Get params for current cluster alg
        params = self.ALGS[self.cluster_alg]

        # Update params with attribute-specific params
        # params['nclust'].extend(self.attr_params.values()) #########################3

        # Create a list of dicts with format param_name: param_value.
        param_combs = []
        combinations_product = product(*params.values())
        # Iterate through all combinations and pass them to the function
        for combination in combinations_product:
            param_combs.append(dict(zip(params.keys(), combination)))

        return param_combs


    def get_nr_authors(self):
        return len(TextsByAuthor(self.language).nr_texts_per_author)


    def cluster(self, **kwargs):
        # Get method name, which is the same as the name of the clustering algorithm
        method = getattr(self, self.cluster_alg)
        clusters = method(**kwargs)

        clusters = Clusters(self.cluster_alg, self.mx, clusters)
        

        if clusters.df['cluster'].nunique() == 1:
            clusters = None
            self.logger.info(f'All data points put into the same cluster.')

        return clusters
        

class SimmxCluster(ClusterBase):
    '''
    Clustering on a similarity matrix
    '''
    ALGS = {
        'hierarchical': {
            'nclust': [5, 10],
            'method': ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'],
            },
        'spectral': {
            'nclust': [5],
            },
        'kmedoids': {
            'nclust': [5],
            },
        'dbscan': {
            'eps': [0.1, 0.3, 0.5, 0.7, 0.9],
            'min_samples': [5, 10, 30],
            },
    }

    if test:
        ALGS = {
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

    def __init__(self, language=None, cluster_alg=None, mx=None):
        super().__init__(language=language, cluster_alg=cluster_alg)
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
        d = DBSCAN(eps=kwargs['eps'], min_samples=kwargs['min_samples'], metric='precomputed', n_jobs=self.n_jobs)
        clusters = d.fit_predict(self.mx.dmx)
        return clusters


class NetworkCluster(ClusterBase):
    ALGS = {'girvan': {
                'noparam': [None], # Girvan, Newman (2002): Community structure in social and biological networks
            },            
            'lpa': {
                'noparam': [None],
            },
            'louvain': {
                'resolution': [100, 10, 1, 0.1, 0.01],
            },
    }

    if test:
        ALGS = {'girvan': {},           
                'lpa': {},
                'louvain': {
                    'resolution': [1],
                },
        }

    def __init__(self, language, cluster_alg, network):
        super().__init__(language, cluster_alg)
        self.network = network
        self.graph = network.graph
        # for edge in self.graph.edges(data=True):
        #     source, target, weight = edge
        #     print(f"Edge: {source} - {target}, Weight: {weight['weight']}")

    def lpa(self, **kwargs):
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
