import os
import logging
logging.basicConfig(level=logging.DEBUG)
from sklearn.utils import shuffle
from multiprocessing import cpu_count
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from pathlib import Path
import sys
sys.path.append("..")
from utils import load_list_of_lines, save_list_of_lines, get_bookname
import time
import collections
from feature_extraction.process import Tokenizer


class Doc2VecChunkVectorizer():
    def __init__(self,
                 language,
                 doc_paths,
                 mode, # doc_tags, chunk_tags, chunk_features
                 tokens_per_chunk=500,
                 dm=1,
                 dm_mean=1,
                 seed=42,
                 n_cores=-1):
        self.language = language
        self.doc_paths = doc_paths
        self.mode = mode
        self.tokens_per_chunk = tokens_per_chunk
        self.dm = dm
        self.dm_mean = dm_mean
        self.seed = seed
        if n_cores == -1 or n_cores is None:
            self.n_cores = cpu_count()-1
        else:
            self.n_cores = n_cores

    def tag_chunks(self):
        tagged_chunks = []
        self.doc_path_to_chunk_ids = {}
        logging.info('Preparing data for Doc2VecChunkVectorizer...')
        for doc_path in self.doc_paths:
            chunk_id_counter = 0

            all_chunks = Tokenizer(self.language).get_tokenized_words(doc_path, remove_punct=True, lower=True)
            for curr_chunks in all_chunks:
                words = curr_chunks.split()
                assert len(words) < 10000 # Doc2vec has token limit of 10'000
                
                if self.mode == 'doc_tags':
                    doc_tag = get_bookname(doc_path)
                    tagged_chunks.append(TaggedDocument(words=words, tags=[doc_tag]))
                elif self.mode == 'both_tags':
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

        logging.info('Prepared data for Doc2VecChunkVectorizer.')
        return tagged_chunks

    def fit(self):
        tagged_chunks = self.tag_chunks()
        logging.info('Fitting Doc2VecChunkVectorizer...')
        self.model = Doc2Vec(shuffle(tagged_chunks, random_state=3), #vector_size=100 by default
                                 window=10,
                                 dm=self.dm,
                                 dm_mean=self.dm_mean,
                                 workers=self.n_cores,
                                 min_count=5,
                                 seed=self.seed)
        logging.info('Fitted Doc2VecChunkVectorizer.')
    
    def transform(self):
        logging.info('Saving chunk vectors...')
        
        try:
            for doc_path in self.doc_paths: 
                if self.mode == 'doc_tags': 
                    dvs = self.model.dv[get_bookname(doc_path)]
                    path = doc_path.replace('/raw_docs', f'/doc2vec_doctags_tpc_{self.tokens_per_chunk}')
                    save_list_of_lines(dvs, path, 'np')

                # elif self.mode == 'chunk_tags':
                #     dvs = self.infer_dv(doc_path)
                #     path = doc_path.replace('/raw_docs', f'/doc2vec_chunktags_tpc_{self.tokens_per_chunk}')
                #     save_list_of_lines(dvs, path, 'np')

                elif self.mode == 'chunk_features':
                    dvs = {chunk_id: self.model.dv[chunk_id] for chunk_id in self.doc_path_to_chunk_ids[doc_path]}
                    path = doc_path.replace('/raw_docs', f'/doc2vec_chunk_embeddings_tpc_{self.tokens_per_chunk}')
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, **dvs)

            if self.mode == 'both_tags':
                dvs = {str(chunk_id): self.model.dv[chunk_id] for chunk_id in self.model.dv.index_to_key}
                path = doc_path.replace('/raw_docs', f'/doc2vec_bothtags_tpc_{self.tokens_per_chunk}')
                path = Path.joinpath(Path(path).parent, 'dv')
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **dvs)
            logging.info('Saved chunk vectors.')
        except NameError as e:
            print(e)
            print('Doc2vec model has not been trained.')

    def assess_chunks(self):
        '''
        Adapted from doc2vec documentation
        Infer new vectors for each document of the training corpus, compare the inferred vectors with the training corpus, 
        and then returning the rank of the document based on self-similarity.
        Checking the inferred-vector against a training-vector is a sort of sanity check as to whether the model is behaving 
        in a usefully consistent manner, though not a real accuracy value.
        '''
        ranks = []
        second_ranks = []
        for doc_path in self.doc_paths:
            chunk_id_counter = 0

            all_chunks = Tokenizer(self.language).get_tokenized_words(doc_path, remove_punct=True, lower=True)
            for curr_chunks in all_chunks:
                # Split text into word list
                doc_tag = f'{get_bookname(doc_path)}_{chunk_id_counter}'  ### Convert to int to save memory ################# use list with two tags instead
                words = curr_chunks.split()
                inferred_vector = self.model.infer_vector(words)
                sims = self.model.dv.most_similar([inferred_vector])
                print(doc_tag, sims)
                rank = [docid for docid, sim in sims].index(doc_tag)
                ranks.append(rank)
                second_ranks.append(sims[1])
                chunk_id_counter += 1
        counter = collections.Counter(ranks)
        print(counter)


    def infer_dv(self, doc_path):
        '''
        Infer document vector of the whole model after training model on individual chunks.
        '''
        all_words = []
        all_chunks = Tokenizer(self.language).get_tokenized_words(doc_path, remove_punct=True, lower=True)
        for chunk in all_chunks:
            chunk = chunk.split()
            all_words.extend(chunk)

        inferred_vector = self.model.infer_vector(all_words)
        return inferred_vector

    def assess_docs(self):
        '''
        Adapted from doc2vec documentation
        Infer new vectors for each document of the training corpus, compare the inferred vectors with the training corpus, 
        and then returning the rank of the document based on self-similarity.
        Checking the inferred-vector against a training-vector is a sort of sanity check as to whether the model is behaving 
        in a usefully consistent manner, though not a real accuracy value.
        '''
        ranks = []
        second_ranks = []
        for doc_path in self.doc_paths:
            inferred_vector = self.infer_dv(doc_path)

            sims = self.model.dv.most_similar([inferred_vector])
            rank = [docid for docid, sim in sims].index(get_bookname(doc_path))
            ranks.append(rank)
            second_ranks.append(sims[1])
        counter = collections.Counter(ranks)
        print(counter)


# f it's not tens-of-thousands of texts, use a smaller vector size and more epochs (but realize results may still be weak with small data sets)
# gensim BLAS
# Assessing the Model
# Tokenizer
