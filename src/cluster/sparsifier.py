import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import sys
import pickle
from itertools import product
import os
sys.path.append("..")
from utils import TextsByAuthor, DataHandler
from .create import SimMx
from .simmelian import Simmelian
from .cluster_utils import CombinationInfo

import logging
logging.basicConfig(level=logging.DEBUG)
# # Suppress logger messages by 'matplotlib.ticker
# # Set the logging level to suppress debug output
# ticker_logger = logging.getLogger('matplotlib.ticker')
# ticker_logger.setLevel(logging.WARNING)

class Sparsifier(DataHandler):
      # MODES = {
      #       'threshold': {
      #             'cutoff': [0.9],
      #             },
      #       'author': [None],
      #       'simmel': {
      #             'overlap_k': [(5, 10), (50, 100)],
      #       },
      # }
      #'passthrough': [None]

      MODES = {
            'threshold': [0.9],
            'authormax': [None],
            'simmel': [(5, 10), (50, 100)],
            }
      '''
      threshold values: the threshold for filtering will be set at a value below which 90% of the values in the matrix fall.
      '''
      def __init__(self, language=None, mx=None, mode=None):
            super().__init__(language, output_dir='similarity')
            self.mx = mx
            self.mode = mode
            self.logger = logging.getLogger(__name__)
            self.add_subdir('sparsification')


      # def get_param_combinations(self):
      #       # Get params for current cluster alg
      #       params = self.MODES[self.mode]

      #       # Create a list of dicts with format param_name: param_value.
      #       param_combs = []
      #       combinations_product = product(*params.values())
      #       # Iterate through all combinations and pass them to the function
      #       for combination in combinations_product:
      #             param_combs.append(dict(zip(params.keys(), combination)))

      #       return param_combs


      def sparsify(self, spars_param):
            info = CombinationInfo(mxname=self.mx.name, sparsmode=self.mode, spars_param=spars_param)
            print(info.as_df())
            pkl_path = self.get_file_path(file_name=f'sparsmx-{info.as_string()}.pkl', subdir=True)
            if os.path.exists(pkl_path):
                  with open(pkl_path, 'rb') as f:
                        simmx = pickle.load(f)
            else:
                  start = time.time()
                  mx = copy.deepcopy(self.mx.mx)
                  if self.mode == 'threshold':
                        mx = self.filter_threshold(mx, spars_param)
                        simmx = SimMx(self.language, name=self.mx.name, mx=mx, normalized=True, is_sim=True, is_directed=False)
                        self.logger.info(f'{self.mode} sparsification: matrix is undirected')

                  elif self.mode == 'authormin':
                        mx = self.filter_author_similarity(mx, by='min')
                        simmx = SimMx(self.language, name=self.mx.name, mx=mx, normalized=True, is_sim=True, is_directed=True)
                        self.logger.info(f'{self.mode} sparsification: matrix is directed')

                  elif self.mode == 'authormax':
                        mx = self.filter_author_similarity(mx, by='max')
                        simmx = SimMx(self.language, name=self.mx.name, mx=mx, normalized=True, is_sim=True, is_directed=True)
                        self.logger.info(f'{self.mode} sparsification: matrix is directed')

                  elif self.mode == 'simmel':
                        mx = self.filter_simmelian(mx, spars_param)
                        simmx = SimMx(self.language, name=self.mx.name, mx=mx, normalized=True, is_sim=True, is_directed=True)
                        self.logger.info(f'{self.mode} sparsification: matrix is directed') # Directed with conditioned=True

                  with open(pkl_path, 'wb') as f:
                        pickle.dump(simmx, f)
                  print(f'{time.time()-start}s to sparsify with {self.mode} {spars_param}.')

            self.plot_degree_dist(simmx, info)

            print(f'Nr vals before filtering: {simmx.mx.shape[0]**2-len(simmx.mx)}.')
            print(f'Nr vals after filtering: {np.count_nonzero(simmx.mx.values)}.')
            return simmx
      
      
      def plot_degree_dist(self, simmx, info):
            # Calculate the degree for each node
            degrees = simmx.mx.sum(axis=1).sort_values()
            plt.figure(figsize=(20, 6))
            plt.bar(range(len(degrees)), degrees, width=0.8)
            plt.title('Degree Distribution')
            plt.xlabel('Node')
            plt.ylabel('Degree')
            plt.grid(False)
            self.save_data(data=plt, file_name=f'degree-dist-{info.as_string()}', subdir=True, data_type='png', plt_kwargs={'dpi': 600})
            plt.close()


      def filter_threshold(self, mx, cutoff):
            vals = self.mx.get_triangular()
            threshold = np.quantile(a=vals, q=cutoff)
            mx[mx<threshold] = 0
            return mx


      def filter_author_similarity(self, mx, by):
            print(by)
            '''
            mx: similarity marix
            Edge only if texts are equally or more similar than the least (if by='min') (most if by='max) similar text by the same author
            This results in a directed weight matrix
            '''
            directed_mx = []
            author_filename_mapping = TextsByAuthor(self.language, filenames=mx.index).author_filename_mapping
            for _, list_of_filenames in author_filename_mapping.items():
                  author_mx = mx.loc[list_of_filenames, list_of_filenames].copy()                
                  new_rows = mx.loc[list_of_filenames].copy() 

                  if by == 'min':
                        if author_mx.shape[0] != 1:
                              autsim = author_mx.min().min()
                              new_rows[new_rows < autsim] = 0
                        else:
                              new_rows[:] = 0
                  else:
                        if author_mx.shape[0] != 1:
                              autsim = np.max(author_mx.values[author_mx.values != 1])
                              new_rows[new_rows < autsim] = 0
                        else:
                              new_rows[:] = 0
                  directed_mx.append(new_rows)

            directed_mx = pd.concat(directed_mx)
            directed_mx = directed_mx.sort_index(axis=0).sort_index(axis=1)
            assert directed_mx.shape == mx.shape
            assert directed_mx.notna().all().all()
            assert directed_mx.shape == self.mx.mx.shape
            return directed_mx


      def filter_simmelian(self, mx, overlap_k):
            min_overlap = overlap_k[0]
            k = overlap_k[1]
            s = Simmelian(mx, conditioned=True)
            mx = s.run_parametric(min_overlap=min_overlap, k=k)
            return mx