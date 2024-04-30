

# %%
import sys
sys.path.append("..")
import time
import os
from node2vec import Node2Vec
from .embedding_utils import EmbeddingBase


class N2vCreator(EmbeddingBase):
      def __init__(self, language, mode=None):
            super().__init__(language, output_dir='n2v', edgelist_dir='sparsification_edgelists', mode=mode)


      def create_embeddings(self, edgelist, embedding_path,  kwargs={}):
            network = self.network_from_edgelist(os.path.join(self.edgelist_dir, edgelist))
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
            model.wv.save_word2vec_format(embedding_path)

            # Save model
            model_filename = edgelist.replace('.csv', '')
            model_filename = os.path.join(self.subdir, f'{edgelist}.model')
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


# %%
