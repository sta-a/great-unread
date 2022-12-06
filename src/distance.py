# %%
%load_ext autoreload
%autoreload 2
import pandas as pd
import os
from scipy.spatial.distance import minkowski
from sklearn_extra.cluster import KMedoids
from matplotlib import pyplot as plt
import sys
from matplotlib import pyplot as plt
sys.path.insert(1, '/home/annina/scripts/pydelta')
import delta
from distance_functions import get_pydelta_mx, ClusterVis, get_mx
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from hpo_functions import get_author_groups, get_data
import matplotlib as mpl
from sklearn.manifold import MDS

languages = ['eng', 'ger']
data_dir = '../data'

# Hoffmansthal_Hugo-von_Ein-Brief_1902 # canon scores
# Hoffmansthal_Hugo_Ein-Brief_1902 # raw docs

# Hegelers_Wilhelm_Mutter-Bertha_1893 # canon scores
# Hegeler_Wilhelm_Mutter-Bertha_1893 # raw docs

# Std of whole corpus or only mfw????
# function registry
# use all distances
# agglomerative hierarchical clustering, k-means, or density-based clustering (DBSCAN)


# %%
class Clustering():
    def __init__(
            self, 
            draw,
            language, 
            dist_name,
            dists,
            group=None, 
            distances_dir = None,
            sentiscores_dir = None,
            metadata_dir = None,
            canonscores_dir = None,
            features_dir = None):

        self.draw = draw
        self.group = group
        self.language = language
        self.dist_name = dist_name
        self.dists=dists
        self.distances_dir = distances_dir
        self.sentiscores_dir = sentiscores_dir
        self.metadata_dir = metadata_dir
        self.canonscores_dir = canonscores_dir
        self.features_dir = features_dir
        self.group_params ={
            'unread': {'n_clust': 2},
            'gender': {'n_clust': 2}}
        self.mx = self.dists[self.dist_name]['mx']
        if self.draw == True:
            self.vis = ClusterVis(
                language=self.language,
                dist_name = self.dist_name,
                dists=self.dists,
                group=self.group,
                distances_dir=self.distances_dir,
                sentiscores_dir=self.sentiscores_dir,
                metadata_dir=self.metadata_dir,
                canonscores_dir=self.canonscores_dir,
                features_dir=self.features_dir)

    def get_clusters(self, type):
        if type == 'hierarchical':
            clustering = self.hierarchical()
        elif type == 'kmedoids':
            clustering = self.kmedoids()
        return clustering

    def kmedoids(self):
        n_clusters = self.group_params[self.group]['n_clust']
        kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', method='pam', init='build', random_state=8)
        clustering = kmedoids.fit_predict(self.mx)
        clustering = pd.DataFrame(clustering, index=self.mx.index).rename({0: 'cluster'}, axis=1)
        if self.draw == True:
            self.vis.draw_mds(clustering)
        return clustering

    def hierarchical(self):
        # from Pydelta
        # Ward

        # Linkage matrix
        clustering = sch.ward(ssd.squareform(self.mx, force="tovector"))
        if self.draw == True:
            self.vis.draw_dendrogram(clustering)
        return clustering




for language in languages:
    distances_dir = os.path.join(data_dir, 'distances', language)
    if not os.path.exists(distances_dir):
        os.makedirs(distances_dir, exist_ok=True)
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    canonscores_dir = os.path.join(data_dir, 'canonscores')
    features_dir = os.path.join(data_dir, 'features_None', language)

    dists = {}
    # dists['imprt'] = { ######################################################33
    #     'mx': get_mx('imprt', language, data_dir),
    #     'nmfw': None}

    pydelta_dists = ['burrows', 'quadratic', 'eder', 'edersimple', 'cosinedelta']
    nmfw_list = [500, 1000, 2000, 5000]
    pydelta_dists = ['quadratic']
    nmfw_list = [500]
    for dist in pydelta_dists:
        for nmfw in nmfw_list:
            print('----------------------------', dist, nmfw)
            dist_name = f'{dist}{nmfw}'
            dists[dist_name] = {
            'mx': get_mx(dist_name, language, data_dir, nmfw=nmfw, function=dist),
            'nmfw': nmfw}

            print(dists[dist_name]['mx'])


    for dist_name in dists.keys():
        for group in ['gender']: #'author', 'unread', 
            c = Clustering(
                draw=True,
                language=language, 
                dist_name=dist_name, 
                dists=dists, #######################3
                group=group,
                distances_dir = distances_dir,
                sentiscores_dir = sentiscores_dir,
                metadata_dir = metadata_dir,
                canonscores_dir = canonscores_dir,
                features_dir = features_dir)
        clusters = c.get_clusters(type='hierarchical')


        for group in ['unread', 'gender']:
            c = Clustering(
                draw=True,
                language=language, 
                dist_name=dist_name, 
                dists=dists,
                group=group,
                distances_dir = distances_dir,
                sentiscores_dir = sentiscores_dir,
                metadata_dir = metadata_dir,
                canonscores_dir = canonscores_dir,
                features_dir = features_dir)
            clusters = c.get_clusters(type='kmedoids')


# %%

# Use setter for goup in clustering constuctor