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
from distances_functions import ImprtDistance, PydeltaDist, is_symmetric, show_distance_distribution

nr_texts = None
language = 'ger' #['eng', 'ger']
data_dir = '../data'

distances_dir = os.path.join(data_dir, 'distances', language)
if not os.path.exists(distances_dir):
    os.makedirs(distances_dir, exist_ok=True)


# %%
###################
# Get distance based on feature importance
###################
i = ImprtDistance(language, data_dir)
imprtmx = i.calculate_dist_matrix(file_name='imprtdist')

# %%
###################
# Get Pydelta distances
###################
#print(delta.functions)
nmfw = 500
pydelta = PydeltaDist(language, data_dir, nmfw=nmfw, nr_texts=nr_texts)
pydelta_corpus = pydelta.get_corpus()

#x = corpus.sum(axis=1).sort_values(ascending=False)
burrowsmx = delta.functions.burrows(pydelta_corpus) #MFW?
print(is_symmetric(burrowsmx))
# %%
show_distance_distribution(burrowsmx, nmfw)






# %%
# kmedoids = KMedoids(n_clusters=2, metric='precomputed', method='pam', init='build', random_state=8).fit(burrowsmx)
from sklearn.manifold import MDS
X_transform = MDS(n_components=2, dissimilarity='precomputed', random_state=8).fit_transform(burrowsmx)


# distance, dissimilarity'
# Std of whole corpus or only mfw????
# function registry
#similarity or dissimiliarity
# use all distances

# agglomerative hierarchical clustering, k-means, or density-based clustering (DBSCAN)

