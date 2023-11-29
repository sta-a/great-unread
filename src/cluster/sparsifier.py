import pandas as pd
import numpy as np
import copy
import time
import sys
sys.path.append("..")
from utils import TextsByAuthor
from .create import SimMx

import logging
logging.basicConfig(level=logging.DEBUG)
# # Suppress logger messages by 'matplotlib.ticker
# # Set the logging level to suppress debug output
# ticker_logger = logging.getLogger('matplotlib.ticker')
# ticker_logger.setLevel(logging.WARNING)

class Sparsifier():
      MODES = {
            'threshold': [0.9],
            'author': [None],
            #'passthrough': [None]
      }

      '''
      threshold values: the threshold for filtering will be set at a value below which 90% of the values in the matrix fall.
      '''
      def __init__(self, language=None, mx=None, mode=None, spars_param=None):
            self.language = language
            self.mx = mx
            self.mode = mode
            self.spars_param = spars_param
            self.threshold_params = [0.9]
            self.logger = logging.getLogger(__name__)

      def set_diagonal(self, mx, value):
            for i in range(0, mx.shape[0]):
                mx.iloc[i, i] = value
            return mx

      def sparsify(self):
            start = time.time()
            mx = copy.deepcopy(self.mx.mx)
            if self.mode == 'threshold':
                  mx = self.filter_threshold(mx)
                  simmx = SimMx(self.language, name=self.mx.name, mx=mx, normalized=True, is_sim=True, is_directed=False)
                  self.logger.info(f'{self.mode} sparsification: matrix is undirected')

            elif self.mode == 'author':
                  mx = self.filter_min_author_similarity(mx)
                  simmx = SimMx(self.language, name=self.mx.name, mx=mx, normalized=True, is_sim=True, is_directed=True)
                  self.logger.info(f'{self.mode} sparsification: matrix is directed')
            elif self.mode == 'passthrough':
                  # Return unchanged matrix
                  simmx = self.simmx
            print(f'{time.time()-start}s to sparsify with {self.mode}-{self.spars_param}.')
            return simmx

      def filter_threshold(self, mx):
            vals = self.mx.get_triangular()
            threshold = np.quantile(a=vals, q=self.spars_param)
            mx[mx<threshold] = 0
            # mx = self.set_diagonal(mx, 0)
            print(f'Threshold: {threshold}.')
            print(f'Nr vals before filtering: {mx.shape[0]**2}.')
            print(f'Nr vals after filtering: {np.count_nonzero(mx.values)}.')
            return mx

      def filter_min_author_similarity(self, mx):
            '''
            mx: similarity marix
            Edge only if texts are equally or more similar than the least similar text by the same author
            This results in a directed weight matrix
            '''
            directed_mx = []
            author_filename_mapping = TextsByAuthor(self.language, list_of_filenames=mx.index).author_filename_mapping
            for _, list_of_filenames in author_filename_mapping.items():
                  author_mx = mx.loc[list_of_filenames, list_of_filenames].copy()
                  #   new_rows.where(mx >= min_similarity, 0)
                  
                  new_rows = mx.loc[list_of_filenames].copy() 
                  if author_mx.shape[0] != 1:
                        min_simliarity = author_mx.min().min()
                        print('min sim sparse', min_simliarity)
                        new_rows[new_rows < min_simliarity] = 0
                  else:
                        new_rows[:] = 0
                  directed_mx.append(new_rows)

            directed_mx = pd.concat(directed_mx)
            directed_mx = directed_mx.sort_index(axis=0).sort_index(axis=1)
            assert directed_mx.shape == mx.shape
            assert directed_mx.notna().all().all()
            assert directed_mx.shape == self.mx.mx.shape
            print(f'Number of possible edges: {directed_mx.shape[0]*(directed_mx.shape[0]-1)}')
            print(f'Number of non-zero: {(directed_mx.ne(0)).sum().sum()}.')

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