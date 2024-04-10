

# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")
import pandas as pd
import time
import pickle
import os
import networkx as nx
import matplotlib as plt
from node2vec import Node2Vec
from utils import DataHandler
from tqdm import tqdm
from embedding_utils import EmbeddingBase


class N2vCreator(EmbeddingBase):
      def __init__(self, language):
            super().__init__(language, output_dir='n2v')


      def create_embeddings(self, fn, kwargs={}):
            network = self.network_from_edgelist(os.path.join(self.edgelist_dir, fn))
            s = time.time()


            # Adapted from https://github.com/eliorc/node2vec
            node2vec = Node2Vec(
                  graph=network, 
                  weight_key='weight',
                  quiet=True,
                  workers=7,
                  **kwargs)

            #      dimensions=64, 
            #       walk_length=8, 
            #       num_walks=200, 
            #       p=1,
            #       q=1,
            # Embed nodes
            model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

            # Save embeddings
            embedding_path = self.get_embedding_path(fn, kwargs)
            model.wv.save_word2vec_format(embedding_path)

            # Save model
            model_filename = os.path.join(self.subdir, f'{fn}.model')
            model.save(model_filename)
            print(f'{time.time()-s}s to get embeddings for one mx.')


      def get_params(self):
            params = {
            'dimensions': [32, 64, 128],
            'walk_length': [8],
            'num_walks': [200],
            'p': [0.5, 1, 2],
            'q': [1]
            }
            return params





# for language in ['eng', 'ger']:
#       ne = N2vCreator(language)
      # ne.get_network_selfloops()
      # ne.analyze_network_stats()
      # ne.create_data()
      # ne.run_combinations()


# %%
