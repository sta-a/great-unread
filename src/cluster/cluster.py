from sklearn_extra.cluster import KMedoids
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS
import pandas as pd
import itertools
from matplotlib import pyplot as plt
import networkx as nx
from networkx import edge_betweenness_centrality as betweenness
from networkx.algorithms.community.centrality import girvan_newman
import os
import sys
sys.path.append("..")

from utils import DataHandler, TextsByAuthor
from .network_viz import MetadataHandler

import logging
logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
ticker_logger = logging.getLogger('matplotlib.ticker')
ticker_logger.setLevel(logging.WARNING)


class ClusterQuality(MetadataHandler):
    '''
    cluster_list: list of filenames or list of lists where each inner list represents one cluster
    '''
    def __init__(
            self,
            language,
            attribute_name, 
            cluster_list):
        super().__init__(language=language, attribute_name=attribute_name)
        self.cluster_list = cluster_list
        self.metadf = self.get_metadata()

    def assess_quality(self):
        if self.attribute_name == 'gender':
            metadf = self.metadf[self.metadf['gender'].isin(['m', 'f'])] # ignore anonymous and both
            print(metadf)
            ratios = []
            if all(isinstance(item, list) for item in self.cluster_list):
                for i in range(0, len(self.cluster_list)):
                    df = metadf[metadf['file_name'].isin(self.cluster_list[i])]
                    counts = df['gender'].value_counts()
                    print(counts)

                    vals_in_col = counts.index.tolist()
                    if 'm' in vals_in_col and 'f' in vals_in_col:
                        mf_ratio = counts['m'] / counts['f']
                    elif 'm' in vals_in_col:
                        mf_ratio = 1
                    else:
                        mf_ratio = 0
                    ratios.append(mf_ratio)
            else:
                raise ValueError('Clusters are not a list of lists.')
            print(f'gender cluster ratios: {ratios}')
        elif self.attribute_name == 'author':
            author_filename_mapping = TextsByAuthor.author_filename_mapping


        else:
            raise NotImplementedError('Cluster Quality only implemented for gender.')
        return ratios


class SimmxCluster():
    '''
    Clustering on a similarity matrix
    '''
    def __init__(self, language=None, mx=None, attribute_name=None, cluster_alg=None):
        self.language = language
        self.mx = mx
        self.attribute_name = attribute_name
        self.cluster_alg = cluster_alg
        # self.group_params ={
        #     'unread': {'n_clusters': 2, 'type': 'kmedoids'},
        #     'gender': {'n_clusters': 2, 'type': 'kmedoids'},
        #     'author': {'type': 'hierarchical'}}
        self.cluster_algs = ['spectral']####################, 'hierarchical', 'kmedoids']
        self.attr_params = {'gender': 2, 'canon': None, 'author': self.get_nr_authors()}

    def get_nr_authors(self):
        return len(TextsByAuthor(self.language).nr_texts_per_author)

    def cluster(self):
        if self.cluster_alg == 'spectral':
            clust = self.spectral()
        elif self.cluster_alg == 'hierarchical':
            clust = self.hierarchical()
        else:
            clust = self.kmedoids()

        if not all(isinstance(item, list) for item in clust):
            clust = self.nr_to_fn(clust)
        self.clusters = clust
        self.qual = ClusterQuality(self.language, self.attribute_name, clust).assess_quality()
    
    def nr_to_fn(self, clust):
        '''
        If result of clustering algorithm is a list with cluster assingments as ints, turn it into a list of lists of file names.
        '''
        assert self.mx.equals(self.mx.T)
        new_clust = []
        for nr in set(clust):
            bool_list = [1 if el == nr else 0 for el in clust]
            bool_list = pd.Series(bool_list).astype(bool)
            fn_list = self.mx.columns[bool_list].tolist()
            new_clust.append(fn_list)
        return new_clust

    def spectral(self):
        clust = SpectralClustering(n_clusters=self.attr_params[self.attribute_name], eigen_solver='arpack', random_state=11, affinity='precomputed', assign_labels='kmeans').fit_predict(self.mx)
        return clust

    def hierarchical(self):
        # from Pydelta
        # Ward

        # Linkage matrix
        clusters = sch.ward(ssd.squareform(self.mx, force="tovector"))
        return clusters
        
    def kmedoids(self, group):
        n_clusters = self.group_params[group]['n_clusters']
        kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', method='pam', init='build', random_state=8)
        clusters = kmedoids.fit_predict(self.mx)
        clusters = pd.DataFrame(clusters, index=self.mx.index).rename({0: 'cluster'}, axis=1)
        return clusters


class NetworkCluster():
    def __init__(self, graph=None, attribute_name=None, cluster_alg=None):
        self.graph = graph
        self.attribute_name = attribute_name
        self.cluster_alg = cluster_alg
        # self.group_params ={
        #     'unread': {'n_clusters': 2, 'type': 'kmedoids'},
        #     'gender': {'n_clusters': 2, 'type': 'kmedoids'},
        #     'author': {'type': 'hierarchical'}}
        self.cluster_algs = ['gn']#######################, 'louvain']
        self.attr_params = {'gender': 2, 'canon': None, 'author': None}
    
    def find_louvain_resolution(self):
        # Does not work for 2 clusters
        n=1000
        resolution = 10
        while n > 2:
            resolution /= 10
            clust = nx.community.louvain_communities(self.graph, weight='weight', seed=111, resolution=resolution)
            n = len(clust)
            print(f'{len(clust)} clusters after Louvain clustering with resolution {resolution}.')
        return clust

    def get_gn(self):

        def most_central_edge(graph):
            # Utilize edge weights when choosing an edge with the highest betweenness centrality
            centrality = betweenness(graph, weight="weight")
            return max(centrality, key=centrality.get)
        
        comp = girvan_newman(self.graph, most_valuable_edge=most_central_edge)
        # limited = itertools.takewhile(lambda clust: len(clust) == self.attr_params[self.attribute_name], comp)
        # Problem: minimum number of clusters bigger than predefined cluster sizes
        for communities in itertools.islice(comp, 1):
            clust = tuple(sorted(clust) for clust in communities)
        print(clust)
        # print('gn results', len(clust))
        # assert len(clust) == self.attr_params[self.attribute_name]
        return clust

    def cluster(self):
        # if self.cluster_alg == 'louvain':
        #     if self.attribute_name == 'gender':
        #         clust = self.find_louvain_resolution()
        if self.cluster_alg == 'louvain':
            clust = nx.community.louvain_communities(self.graph, weight='weight', seed=111, resolution=1)
        elif self.cluster_alg == 'gn':
            clust = self.get_gn()
        self.clusters = clust




# class SimmxClusterViz(DataHandler):
#     def __init__(
#             self, 
#             language,
#             dist_name,
#             mx,
#             group, 
#             metadata_dir,
#             canonscores_dir):
#         super().__init__(language, 'distance')
    
#         self.dist_name = dist_name
#         self.mx = mx
#         self.group = group
#         self.metadata_dir = metadata_dir
#         self.canonscores_dir = canonscores_dir
#         self.file_group_mapping = self._init_colormap()
#         self.plot_name = f'{self.dist_name}_{self.group}_{self.language}'

#     def _relabel_axis(self):
#         labels = self.ax.get_ymajorticklabels()
#         for label in labels:
#             color = self.file_group_mapping.loc[self.file_group_mapping['file_name'] ==label.get_text(), 'group_color']
#             label = label.set_color(str(color.values[0]))

#     def save(self,plt, vis_type, dpi):
#         plt.savefig(os.path.join(self.output_dir, f'{self.plot_name}_{vis_type}.svg'), dpi=dpi)

#     def draw_dendrogram(self, clusters):
#         print(f'Drawing dendrogram.')
#         plt.clf()
#         plt.figure(figsize=(12,12),dpi=1000)
#         dendro_data = sch.dendrogram(
#             Z=clusters, 
#             orientation='left', 
#             labels=self.mx.index.to_list(),
#             show_leaf_counts=True,
#             leaf_font_size=1)
#         self.ax = plt.gca() 
#         self._relabel_axis()
#         plt.title = self.plot_name
#         #plt.xlabel('Samples')
#         #plt.ylabel('Euclidean distances')
#         self.save(plt, 'hierarchical-dendrogram', 1000)

#     def draw_mds(self, clusters):
#         print(f'Drawing MDS.')
#         df = MDS(n_components=2, dissimilarity='precomputed', random_state=6, metric=True).fit_transform(self.mx)
#         df = pd.DataFrame(df, columns=['comp1', 'comp2'], index=self.mx.index)
#         df = df.merge(self.file_group_mapping, how='inner', left_index=True, right_on='file_name', validate='one_to_one')
#         df = df.merge(clusters, how='inner', left_on='file_name', right_index=True, validate='1:1')

#         def _group_cluster_color(row):
#             color = None
#             if row['group_color'] == 'b' and row['cluster'] == 0:
#                 color = 'darkblue'
#             elif row['group_color'] == 'b' and row['cluster'] == 1:
#                 color = 'royalblue'
#             elif row['group_color'] == 'r' and row['cluster'] == 0:
#                 color = 'crimson'
#             #elif row['group_color'] == 'r' and row['cluster'] == 0:
#             else:
#                 color = 'deeppink'
#             return color

#         df['group_cluster_color'] = df.apply(_group_cluster_color, axis=1)


#         fig = plt.figure(figsize=(5,5))
#         ax = fig.add_subplot(1,1,1)
#         plt.scatter(df['comp1'], df['comp2'], color=df['group_cluster_color'], s=2, label="MDS")
#         plt.title = self.plot_name
#         self.save(plt, 'kmedoids-MDS', dpi=500)


