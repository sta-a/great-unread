import pandas as pd
import numpy as np
from .distance_analysis import get_mx_triangular

def filter_threshold(mx, q):
      vals = get_mx_triangular(mx)
      threshold_value = np.quantile(a=vals, q=q)
      mx[mx<threshold_value] = 0
      print(f'Nr vals before filtering: {mx.shape[0]**2}. Nr vals after filtering: {np.count_nonzero(mx.values)}.')
      return mx


def filter_min_author_similarity(mx):
      '''
      mx: similarity marix
      Edge only if texts are equally or more similar than the least similar text by the same author
      This results in a directed weight matrix
      '''
      directed_mx = []
      author_filename_mapping, _ = get_texts_by_author(list_of_filenames=mx.index)
      for _, list_of_filenames in author_filename_mapping.items():
            author_mx = mx.loc[list_of_filenames, list_of_filenames]
            new_rows = mx.loc[list_of_filenames]
            if author_mx.shape[0] != 1:
                  min_simliarity = np.nanmin(author_mx.to_numpy())
                  # set all distances above max to nan
                  new_rows[new_rows < min_simliarity] = np.nan
            else:
                  new_rows[:] = np.nan
            directed_mx.append(new_rows)

      directed_mx = pd.concat(directed_mx)
      assert directed_mx.shape == mx.shape

      print(f'Number of possible edges: {directed_mx.shape[0]*(directed_mx.shape[0]-1)}')
      print(f'Number of non-nan entries: {np.sum(directed_mx.count())}.')

      return directed_mx


# def filter_simmelian(mx):
#     '''
#     mx: similarity matrix
#     '''
#     G = nk_from_adjacency(mx)
#     print(nk.overview(G))
#     G.indexEdges()
#     targetRatio = 0.2
#     ## Non-parametric
#     simmelianSparsifier = nk.sparsification.SimmelianSparsifierNonParametric()
#     simmelieanGraph = simmelianSparsifier.getSparsifiedGraphOfSize(G, targetRatio) # Get sparsified graph
#     print('Nr edges before and after filtering', G.numberOfEdges(), simmelieanGraph.numberOfEdges())
#     x = simmelianSparsifier.scores(G)
#     # same nr edges after sparsification - weights not considered?