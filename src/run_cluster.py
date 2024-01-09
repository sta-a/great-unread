# %%
# %load_ext autoreload
# %autoreload 2
# Don't display plots
# %matplotlib agg

import sys
sys.path.append("..")
import os
import argparse

from cluster.combinations import MxCombinations, NkCombinations
from helpers import remove_directories

import logging
logging.basicConfig(level=logging.DEBUG)





logfiles = ['/home/annina/scripts/great_unread_nlp/data/similarity/eng/log_clst.txt']
for i in logfiles:
    if os.path.exists(i):
        os.remove(i)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--language', type=str)
    parser.add_argument('--mode', type=str)

    args = parser.parse_args()

    language = args.language
    mode = args.mode

    print(f"Selected language: {language}")
    print(f"Selected mode: {mode}")



    if mode == 'mxc':
        mx = MxCombinations(language=language)
        mx.create_combinations()
        remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxeval'])
        remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxviz'])
    elif mode == 'mxtop':
        mx = MxCombinations(language=language)
        mx.viz_topk()
    elif mode == 'mxattr':
        mx = MxCombinations(language=language)
        mx.viz_attrs()
        
    elif mode == 'nkc':
        nkc = NkCombinations(language=language)
        nkc.create_combinations()
        remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkeval'])
        remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkviz'])
    elif mode == 'nktop':
        nkc = NkCombinations(language=language)
        nkc.viz_topk()
    elif mode == 'nkattr':
        nkc = NkCombinations(language=language)
        nkc.viz_attrs()




# Elbow for internal cluster evaluation

# Hierarchical clustering:
# From scipy documentation, ignored here: Methods ‘centroid’, ‘median’, and ‘ward’ are correctly defined only if Euclidean pairwise metric is used. If y is passed as precomputed pairwise distances, then it is the user’s responsibility to assure that these distances are in fact Euclidean, otherwise the produced result will be incorrect.

# # # Similarity Graphs (Luxburg2007)
# # eta-neighborhodd graph
# # # find eta
# # eta = 0.1
# set all values below eta to 0

# %%
