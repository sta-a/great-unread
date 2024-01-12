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
      MODES = {
            'threshold': [0.9, 0.95],
            'simmel': [(5, 10), (3, 10), (4,6)], #(7, 10)
            'authormax': [None],
            #'authormin': [None],
            }
      
      def __init__(self, language=None, mx=None, mode=None):
            super().__init__(language, output_dir='similarity')
            self.mx = mx
            self.mode = mode
            self.logger = logging.getLogger(__name__)
            self.add_subdir('sparsification')

      @classmethod
      def load_pkl(cls, spmx_path):
            with open(spmx_path, 'rb') as f:
                  simmx = pickle.load(f)
                  return simmx


      def sparsify(self, spars_param):
            info = CombinationInfo(mxname=self.mx.name, sparsmode=self.mode, spars_param=spars_param)
            spmx_path = self.get_file_path(file_name=f'sparsmx-{info.as_string()}.pkl', subdir=True)
            if os.path.exists(spmx_path):
                  simmx = self.load_pkl(spmx_path)

            else:
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

                  with open(spmx_path, 'wb') as f:
                        pickle.dump(simmx, f)

            self.plot_degree_dist(simmx, info)

            original_nr_edges = simmx.mx.shape[0]**2-len(simmx.mx)
            filtered_nr_edges = np.count_nonzero(simmx.mx.values)
            print(f'Nr vals before filtering: {original_nr_edges}.')
            print(f'Nr vals after filtering: {filtered_nr_edges}.')
            return simmx, filtered_nr_edges, spmx_path
      
      
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
            # With threshold x, x % of edges will be removed.
            vals = self.mx.get_triangular()
            threshold = np.quantile(a=vals, q=cutoff)
            mx[mx<threshold] = 0
            return mx


      def filter_author_similarity(self, mx, by):
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
            # Pickle because calculations take a long time (30s)
            min_overlap = overlap_k[0]
            k = overlap_k[1]
            s = Simmelian(mx, conditioned=True)
            mx = s.run_parametric(min_overlap=min_overlap, k=k)
            return mx