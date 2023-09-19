# %%
import logging
import os
import sys
from collections import Counter
from multiprocessing import cpu_count
from pathlib import Path
import re
from tqdm import tqdm
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
from utils import get_filename_from_path, get_doc_paths, DataHandler, save_list_of_lines
logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.WARNING) # suppress SentenceTransformers batches output 


class RewriteSbertData(DataHandler):
    '''
    Class for writing sbert embeddings that were stored per chunk to single file entry.
    '''
    def __init__(self, language, tokens_per_chunk):
        sbert_output_dir = f'sbert_embeddings'
        super().__init__(language, output_dir=sbert_output_dir, data_type='npz', tokens_per_chunk=tokens_per_chunk)
        self.sb = SbertProcessor(self.language, tokens_per_chunk=500)

    def process_and_save_data(self):
        for doc_path in tqdm(self.doc_paths):
            embeddings = []
            sbert_embeddings = self.sb.load_data(file_name=get_filename_from_path(doc_path), doc_path=doc_path)
            for chunk_idx in sbert_embeddings.keys():
                embeddings.extend(sbert_embeddings[chunk_idx])
            assert all(isinstance(item, np.ndarray) for item in embeddings)
            self.save_data(file_name=get_filename_from_path(doc_path), data=embeddings)
            original_path = os.path.join(self.output_dir, f'{get_filename_from_path(doc_path)}.{self.data_type}').replace('tpc_1000', 'tpc_500')
            print(original_path)
            assert os.path.exists(original_path)
            if os.path.exists(original_path):
                os.remove(original_path)


class SbertProcessor(DataHandler):
    def __init__(self, language, tokens_per_chunk, data_type='npz'):
            super().__init__(language, output_dir='sbert_embeddings', data_type=data_type, tokens_per_chunk=tokens_per_chunk)

            if self.language == 'eng':
                self.sentence_encoder = SentenceTransformer('all-mpnet-base-v2')
            elif self.language == 'ger': # paraphrase-xlm-r-multilingual-v1
                self.sentence_encoder = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')


    def create_data(self, doc_path):
        # print(f'Creating data for {doc_path}')
        chunks = ChunkHandler(self.language, self.tokens_per_chunk).load_data(file_name=get_filename_from_path(doc_path), remove_punct=False, lower=False, as_chunk=True, as_sent=True, doc_path=doc_path)
        
        start = time.time()
        all_embeddings = []
        for chunk in chunks:
            embeddings = list(self.sentence_encoder.encode(chunk))
            all_embeddings.extend(embeddings)
        self.logger.info('\nSaving embeddings.\n')
        self.save_data(file_name=get_filename_from_path(doc_path), data=all_embeddings)
        print(f'Time for {doc_path}: {time.time()-start}')


    def create_all_data(self):
        for doc_path in self.doc_paths:
            _ = self.load_data(file_name=get_filename_from_path(doc_path), load=False, doc_path=doc_path)

    def load_data_type(self, file_path, **kwargs):
        data = np.load(file_path)['arr_0'] # List of arrays
        print('format of sbert data: ', type(data))
        data = np.array(data)
        self.logger.info(f'Returning sbert embedding as an array of arrays.')
        return data


    def check_data(self):
        # Check if there is one embedding for every sentence
        ch = ChunkHandler(self.language, self.tokens_per_chunk)
        _, sentences_per_doc = ch.DataChecker(self.language, ch.output_dir).count_sentences_per_chunk()
        for doc_path in self.doc_paths:
            bookname = get_filename_from_path(doc_path)
            nr_sents = sentences_per_doc[bookname]
            sbert = self.load_data(file_name=f'{bookname}.npz')
            assert len(sbert) == nr_sents
            self.logger.info(f'One embedding per sentence for {bookname}.')


# %%
class D2vProcessor(DataHandler):
    '''
    mode: 
        doc_paths (tag each chunk with document name, use document vector associated with tag)
        both (use both chunk id tags and document name tag as a  list)
        chunk (save document vectors for each chunk to use them as features for prediction)

    Tag each chunk with unique chunk id, infer document vectors: Doesn't work because token limit of 10'000 also applies to inferred documents.
    https://github.com/RaRe-Technologies/gensim/issues/2583
    As infer_vector() uses the same optimized Cython functions as training behind-the-scenes, it also suffers from the same fixed-token-buffer size as training, where texts with more than 10000 tokens have all overflow tokens ignored.

    '''
    def __init__(self, language, tokens_per_chunk, dm=1, dm_mean=1, seed=42, n_cores=-1):
        d2v_output_dir = f'd2v_tpc_{tokens_per_chunk}'
        super().__init__(language, output_dir=d2v_output_dir, data_type='npz', tokens_per_chunk=tokens_per_chunk)

        self.dm = dm
        self.dm_mean = dm_mean
        self.seed = seed
        if n_cores == -1 or n_cores is None:
            self.n_cores = cpu_count()-1
        else:
            self.n_cores = n_cores

        self.modes = ['full', 'chunk', 'both']



    def create_all_data(self):
        # Check if file exists and create it if necessary
        for mode in self.modes:
            print(mode)
            _ = self.create_data(load=False, mode=mode)


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
            bookname = get_filename_from_path(doc_path)
            chunk_idx_counter = 0

            chunks = ChunkHandler(self.language, self.tokens_per_chunk).load_data(file_name=bookname, remove_punct=True, lower=True, as_chunk=True, doc_path=doc_path)
            for chunk in chunks:
                words = chunk.split()
                assert len(words) < 10000 # D2v has token limit of 10'000
                
                if mode == 'full':
                    doc_tag = bookname
                    tagged_chunks.append(TaggedDocument(words=words, tags=[doc_tag]))
                elif mode == 'both':
                    doc_tag = [bookname, f'{bookname}_{chunk_idx_counter}']
                    tagged_chunks.append(TaggedDocument(words=words, tags=doc_tag))
                elif mode == 'chunk':
                    doc_tag = f'{bookname}_{chunk_idx_counter}'
                    tagged_chunks.append(TaggedDocument(words=words, tags=[doc_tag]))
                
                if doc_path in self.doc_path_to_chunk_ids.keys():
                    self.doc_path_to_chunk_ids[doc_path].append(doc_tag)
                else:
                    self.doc_path_to_chunk_ids[doc_path] = [doc_tag]
                chunk_idx_counter += 1

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
        loaded_data = np.load(file_path)
        loaded_dict = {key: loaded_data[key] for key in loaded_data.files}
        data = loaded_dict
        return data
    

    def save_data_type(self, data, file_path, **kwargs):
        logging.info('\nSaving chunk vectors...\n')
        mode = kwargs['mode']
        self.add_subdir(mode)
        dvs = {str(chunk_idx): self.model.dv[chunk_idx] for chunk_idx in self.model.dv.index_to_key}
        print('dv keys', dvs.keys())

        if mode == 'chunk' or mode == 'both':
            ch = ChunkHandler(self.language, self.tokens_per_chunk)
            nr_chunks_per_doc, total_nr_chunks = ch.DataChecker(self.language, ch.output_dir).count_chunks_per_doc()
            
            if mode == 'chunk':
                assert len(dvs) == total_nr_chunks ########################

            for doc_path in self.doc_paths:
                doc_dvs = {}
                bookname = get_filename_from_path(doc_path)
                for chunk_idx in range(0, nr_chunks_per_doc[bookname]):
                    chunkname = f'{bookname}_{chunk_idx}'
                    doc_dvs[chunkname] = dvs[chunkname]
                assert len(doc_dvs) == nr_chunks_per_doc[bookname]

                # mode=both: DVs for both chunks and whole doc, e.g. 'Ainsworth_William-Harrison_Rookwood_1834' and 'Ainsworth_William-Harrison_Rookwood_1834_1'
                if mode == 'both':
                    doc_dvs[bookname] = dvs[bookname]
                file_path = self.get_file_path(file_name=bookname, subdir=True)
                np.savez_compressed(file_path, **doc_dvs)

        else:
            for doc_path in self.doc_paths:
                doc_dvs = {}
                bookname = get_filename_from_path(doc_path)
                doc_dvs[bookname] = dvs[bookname]
                file_path = self.get_file_path(file_name=bookname, subdir=True)
                np.savez_compressed(file_path, **doc_dvs)

        self.model.save(os.path.join(self.output_dir, f'{mode}.model'))
        logging.info('Saved chunk vectors.')


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
    #         chunk_idx_counter = 0

    #         chunks = ChunkHandler(self.language).load_data(doc_path, remove_punct=True, lower=True)
    #         for chunk in chunks:
    #             # Split text into word list
    #             doc_tag = f'{get_filename_from_path(doc_path)}_{chunk_idx_counter}'  ### Convert to int to save memory ################# use list with two tags instead
    #             words = chunk.split()
    #             inferred_vector = self.model.infer_vector(words)
    #             sims = self.model.dv.most_similar([inferred_vector])
    #             print(doc_tag, sims)
    #             rank = [docid for docid, sim in sims].index(doc_tag)
    #             ranks.append(rank)
    #             second_ranks.append(sims[1])
    #             chunk_idx_counter += 1
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
    #         rank = [docid for docid, sim in sims].index(get_filename_from_path(doc_path))
    #         ranks.append(rank)
    #         second_ranks.append(sims[1])
    #     counter = Counter(ranks)
    #     print(counter)



# %%
