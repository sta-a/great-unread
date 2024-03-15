

# %%
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append("..")
import pandas as pd
import pickle
import os
import networkx as nx
import matplotlib as plt
from node2vec import Node2Vec
from utils import DataHandler


class N2VCreator(DataHandler):
      '''
      Enable n2v conda environment.!
      This class creates node2vec embeddings for the sparsified matrices.
      '''
      def __init__(self, language):
            super().__init__(language, output_dir='n2v')
            self.edgelist_dir = os.path.join(self.output_dir, 'sparsification_edgelists')
            self.edgelists = [file for file in os.listdir(self.edgelist_dir) if file.startswith('edgelist') and file.endswith('.csv')]
            self.add_subdir('embeddings')


      def print_graph_info(self, graph):
            print(f'--------------\n'
                  f'nx Graph Overview: \n'
                  f'Nr. of nodes: {graph.number_of_nodes()} \n'
                  f'Nr. of edges: {graph.number_of_edges()} \n'
                  f'is weighted: {nx.is_weighted(graph)} \n'
                  f'is directed: {nx.is_directed(graph)} \n'
                  f'Nr. of Selfloops: {nx.number_of_selfloops(graph)} \n'
                  f'--------------\n')


      def network_from_edgelist(self, file_path, directed):
            if directed == 'directed':
                  graph = nx.read_weighted_edgelist(file_path, delimiter=',', create_using=nx.DiGraph()) ###### set node type to int???
            else:
                  graph = nx.read_weighted_edgelist(file_path, delimiter=',')
            graph.remove_edges_from(nx.selfloop_edges(graph))
            self.print_graph_info(graph)
            return graph


      def extract_directed(self, filename):
            filename = filename.replace('edgelist_', '')
            info, directed = filename.rsplit('_', maxsplit=1)
            directed = directed.split('.')[0]
            print(info, directed)
            return info, directed
      

      def create_data(self):
            for file in self.edgelists:
                  info, directed = self.extract_directed(file)
                  network = self.network_from_edgelist(os.path.join(self.edgelist_dir, file), directed)
                  self.create_embeddings(network, info)


      def create_embeddings(self, network, info):
            # Adapted from https://github.com/eliorc/node2vec
            
            node2vec = Node2Vec(network, dimensions=64, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs

            # Embed nodes
            model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

            # Save embeddings
            EMBEDDING_FILENAME = os.path.join(self.subdir, f'{info}.embeddings')
            model.wv.save_word2vec_format(EMBEDDING_FILENAME)

            # Save model
            EMBEDDING_MODEL_FILENAME = os.path.join(self.subdir, f'{info}.model')
            model.save(EMBEDDING_MODEL_FILENAME)


      def load_embeddings(self, file_name):
            index_mapping = pd.read_csv(os.path.join(self.edgelist_dir, 'index-mapping.csv'), header=0)

            inpath = os.path.join(self.subdir, f'{file_name}.embeddings')
            df = pd.read_csv(inpath, skiprows=1, sep=' ')
            
            # Use the first column with the ID as index
            df = df.set_index(df.columns[0])

            # Rename columns as col1, col2, ...
            df.columns = [f"col{i}" for i in range(1, len(df.columns) + 1)]

            # Map int IDs to file names
            df = df.merge(index_mapping, left_index=True, right_on='new_index', validate='1:1')
            df = df.set_index('original_index')
            df = df.drop('new_index', axis=1)
            df.index = df.index.rename('file_name')
            return df
      

if '__name__' == '__main__':
      ne = N2VCreator('eng')
      ne.create_data()

# %%
