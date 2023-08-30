# %%
import logging
import os
import sys
from collections import Counter
from multiprocessing import cpu_count
from pathlib import Path
import re
import time

import numpy as np
import multiprocessing
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer


from .process_rawtext import ChunkHandler
sys.path.append("..")
from utils import get_bookname, get_doc_paths, DataHandler, get_filename_from_path, save_list_of_lines
logging.basicConfig(level=logging.DEBUG)


class SbertProcessor(DataHandler):

    ### Add multiprocessing
    
    def __init__(self, language, data_type='npz', tokens_per_chunk=500):
        sbert_output_dir = f'sbert_sentence_embeddings_tpc_{tokens_per_chunk}'
        super().__init__(language, output_dir=sbert_output_dir, data_type=data_type, tokens_per_chunk=tokens_per_chunk)


        self.terminating_chars = r'\. | \: | \; | \? | \! | \) | \] | \...'
        if self.language == 'eng':
            self.sentence_encoder = SentenceTransformer('stsb-mpnet-base-v2')
        elif self.language == 'ger':
            self.sentence_encoder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')


    def create_data(self, doc_path):
        print(doc_path)
        chunks = ChunkHandler(self.language, self.doc_paths, self.tokens_per_chunk).load_data(doc_path, remove_punct=False, lower=False, as_chunk=True)

        all_embeddings = []
        for chunk in chunks:
            sentences = re.split(self.terminating_chars, chunk)        
            embeddings = list(self.sentence_encoder.encode(sentences))
            all_embeddings.append(embeddings)
        self.save_data(file_name=get_filename_from_path(doc_path), data=all_embeddings)


    def create_all_data(self):
        for doc_path in self.doc_paths:
            self.file_exists_or_create(file_name=get_filename_from_path(doc_path), doc_path=doc_path)


    def save_data_type(self, data, file_path, **kwargs):
        # data = list of embeddings for each chunk
        logging.info('\nSaving embeddings.\n')
        data = {str(i): data[i] for i in range(0, len(data))}
        np.savez_compressed(file_path, data)
        logging.info('Saved chunk vectors.')

# %%
class D2vProcessor(DataHandler):
    '''
    mode: 
        doc_paths (tag each chunk with document name, use document vector associated with tag)
        both_tags (use both chunk id tags and document name tag as a  list)
        chunk_features (save document vectors for each chunk to use them as features for prediction)

    Tag each chunk with unique chunk id, infer document vectors: Doesn't work because token limit of 10'000 also applies to inferred documents.
    https://github.com/RaRe-Technologies/gensim/issues/2583
    As infer_vector() uses the same optimized Cython functions as training behind-the-scenes, it also suffers from the same fixed-token-buffer size as training, where texts with more than 10000 tokens have all overflow tokens ignored.

    '''
    def __init__(self, language, data_type='npz', tokens_per_chunk=500, dm=1, dm_mean=1, seed=42, n_cores=-1):
        d2v_output_dir = f'd2v_tpc_{tokens_per_chunk}'
        super().__init__(language, output_dir=d2v_output_dir, data_type=data_type, tokens_per_chunk=tokens_per_chunk)

        self.dm = dm
        self.dm_mean = dm_mean
        self.seed = seed
        if n_cores == -1 or n_cores is None:
            self.n_cores = cpu_count()-1
        else:
            self.n_cores = n_cores

        # self.modes = ['doc_tags', 'both_tags', 'chunk_features']


    def create_data(self, **kwargs):
        mode = kwargs['mode']
        start = time.time()
        self.fit(mode)
        self.save_data(data=None, file_name=None, mode=mode)
        print(f'\n{time.time()-start}s for calculating docvecs for {mode}.')


    def tag_chunks(self, mode):
        tagged_chunks = []
        self.doc_path_to_chunk_ids = {}
        logging.info('\nPreparing data for D2vModel...\n')
        for doc_path in self.doc_paths:
            chunk_id_counter = 0

            all_chunks = ChunkHandler(self.language).load_data(doc_path, remove_punct=True, lower=True, as_chunk=True)
            for curr_chunks in all_chunks:
                words = curr_chunks.split()
                assert len(words) < 10000 # D2v has token limit of 10'000
                
                if mode == 'doc_tags':
                    doc_tag = get_bookname(doc_path)
                    tagged_chunks.append(TaggedDocument(words=words, tags=[doc_tag]))
                elif mode == 'both_tags':
                    doc_tag = [get_bookname(doc_path), f'{get_bookname(doc_path)}_{chunk_id_counter}']
                    tagged_chunks.append(TaggedDocument(words=words, tags=doc_tag))
                else:
                    doc_tag = f'{get_bookname(doc_path)}_{chunk_id_counter}'
                    tagged_chunks.append(TaggedDocument(words=words, tags=[doc_tag]))
                
                if doc_path in self.doc_path_to_chunk_ids.keys():
                    self.doc_path_to_chunk_ids[doc_path].append(doc_tag)
                else:
                    self.doc_path_to_chunk_ids[doc_path] = [doc_tag]
                chunk_id_counter += 1

        logging.info('\nPrepared data for D2vModel.\n')
        return tagged_chunks

    def fit(self, mode):
        tagged_chunks = self.tag_chunks(mode)
        self.model = Doc2Vec(shuffle(tagged_chunks, random_state=3), #vector_size=100 by default
                                 window=10,
                                 dm=self.dm,
                                 dm_mean=self.dm_mean,
                                 workers=self.n_cores,
                                 min_count=5,
                                 seed=self.seed)
        logging.info('\nFitted D2vModel.\n')

    
    def load_data_type(self, file_path, **kwargs):
        dvs = np.load(file_path)
        # data = {} # too inefficient for chunks, only for doc_paths
        # for key in list(dvs.keys()):
        #     data[key] = dvs[key]
        data = dvs
        return data
    

    def save_data_type(self, data, file_path, **kwargs):
        logging.info('\nSaving chunk vectors...\n')
        
        try:
            dvs = {str(chunk_id): self.model.dv[chunk_id] for chunk_id in self.model.dv.index_to_key}
            np.savez_compressed(file_path, **dvs)
            self.model.save(str(file_path) + '.model')
            logging.info('Saved chunk vectors.')
        except NameError as e:
            print(e)
            print('\nD2v model has not been trained.\n')

    # def assess_chunks(self):
    #     '''
    #     Adapted from d2v documentation
    #     Infer new vectors for each document of the training corpus, compare the inferred vectors with the training corpus, 
    #     and then returning the rank of the document based on self-similarity.
    #     Checking the inferred-vector against a training-vector is a sort of sanity check as to whether the model is behaving 
    #     in a usefully consistent manner, though not a real accuracy value.
    #     '''
    #     ranks = []
    #     second_ranks = []
    #     for doc_path in self.doc_paths:
    #         chunk_id_counter = 0

    #         all_chunks = ChunkHandler(self.language).load_data(doc_path, remove_punct=True, lower=True)
    #         for curr_chunks in all_chunks:
    #             # Split text into word list
    #             doc_tag = f'{get_bookname(doc_path)}_{chunk_id_counter}'  ### Convert to int to save memory ################# use list with two tags instead
    #             words = curr_chunks.split()
    #             inferred_vector = self.model.infer_vector(words)
    #             sims = self.model.dv.most_similar([inferred_vector])
    #             print(doc_tag, sims)
    #             rank = [docid for docid, sim in sims].index(doc_tag)
    #             ranks.append(rank)
    #             second_ranks.append(sims[1])
    #             chunk_id_counter += 1
    #     counter = Counter(ranks)
    #     print(counter)


    # def infer_dv(self, doc_path):
    #     '''
    #     Infer document vector of the whole model after training model on individual chunks.
    #     '''
    #     all_words = ChunkHandler(self.language).load_data(doc_path, remove_punct=True, lower=True, as_chunk=False)
    #     assert len(all_words) < 10000 # token limit

    #     inferred_vector = self.model.infer_vector(all_words)
    #     return inferred_vector

    # def assess_docs(self):
    #     '''
    #     Adapted from d2v documentation
    #     Infer new vectors for each document of the training corpus, compare the inferred vectors with the training corpus, 
    #     and then returning the rank of the document based on self-similarity.
    #     Checking the inferred-vector against a training-vector is a sort of sanity check as to whether the model is behaving 
    #     in a usefully consistent manner, though not a real accuracy value.
    #     '''
    #     ranks = []
    #     second_ranks = []
    #     for doc_path in self.doc_paths:
    #         inferred_vector = self.infer_dv(doc_path)

    #         sims = self.model.dv.most_similar([inferred_vector])
    #         rank = [docid for docid, sim in sims].index(get_bookname(doc_path))
    #         ranks.append(rank)
    #         second_ranks.append(sims[1])
    #     counter = Counter(ranks)
    #     print(counter)



# %%
