# %%
%load_ext autoreload
%autoreload 2
import pandas as pd
import os
from distance_functions import get_mx, get_clustering
import time

data_dir = '../data'

# Hoffmansthal_Hugo-von_Ein-Brief_1902 # canon scores
# Hoffmansthal_Hugo_Ein-Brief_1902 # raw docs

# Hegelers_Wilhelm_Mutter-Bertha_1893 # canon scores
# Hegeler_Wilhelm_Mutter-Bertha_1893 # raw docs

# Std of whole corpus or only mfw????
# function registry
# use all distances
# agglomerative hierarchical clustering, k-means, or density-based clustering (DBSCAN)


clustering = True
for language in ['eng', 'ger']: #, 'ger'
    distances_dir = os.path.join(data_dir, 'distances', language)
    if not os.path.exists(distances_dir):
        os.makedirs(distances_dir, exist_ok=True)
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    canonscores_dir = os.path.join(data_dir, 'canonscores')
    features_dir = os.path.join(data_dir, 'features_None', language)


    # dists['imprt'] = {
    #     'mx': get_mx('imprt', language, data_dir),
    #     'nmfw': None}

    # pydelta_dists = ['quadratic', 'eder', 'edersimple', 'cosinedelta', 'burrows'] ########################################3
    # nmfw_list = [500, 1000, 2000, 5000]
    pydelta_dists = ['burrows'] 
    nmfw_list = [500]
    for dist in pydelta_dists:
        for nmfw in nmfw_list:
            print('----------------------------', dist, nmfw)
            dist_name = f'{dist}{nmfw}'
            mx = get_mx(language, data_dir, dist_name=dist_name, nmfw=nmfw, function=dist)
            
            if clustering == True:
                get_clustering(
                    draw=True,
                    language=language, 
                    dist_name=dist_name, 
                    mx=mx,
                    distances_dir = distances_dir,
                    sentiscores_dir = sentiscores_dir,
                    metadata_dir = metadata_dir,
                    canonscores_dir = canonscores_dir,
                    features_dir = features_dir)


# %%
