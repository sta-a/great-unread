# %%
%load_ext autoreload
%autoreload 2

import os
import pickle
import numpy as np
import pandas as pd
import shutil
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from feature_extraction.doc_based_feature_extractor import DocBasedFeatureExtractor
from feature_extraction.corpus_based_feature_extractor import CorpusBasedFeatureExtractor
from feature_extraction.ngrams import NgramCounter

import sys
sys.path.append("..")
from utils import DataHandler, get_filename_from_path


class FeatureProcessor(DataHandler):
    def __init__(self, language):
        super().__init__(language, 'features', test=True) #####################
        self.doc_paths = self.doc_paths[:None]
        self.pickle_dir = self.text_raw_dir.replace('/text_raw', '/pickle')
        os.makedirs(self.pickle_dir, exist_ok=True) 

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            features = pickle.load(f)
            return features
    
    def save_pickle(self, path, datatuple):
        with open(path, 'wb') as f:
            pickle.dump(datatuple, f)
    
    def get_doc_features_helper(self, doc_path, as_chunk, ngrams):
        fe = DocBasedFeatureExtractor(language=self.language, doc_path=doc_path, as_chunk=as_chunk, tokens_per_chunk=self.tokens_per_chunk, ngrams=ngrams)
        chunk_features, book_features = fe.get_all_features()
        return chunk_features, book_features
    

    def get_doc_features(self, as_chunk):
        all_chunk_features = []
        all_book_features = []

        # Load ngrams only once for efficiency
        nc = NgramCounter(self.language)
        ngrams = nc.load_all_ngrams(as_chunk=as_chunk)

        for doc_path in self.doc_paths:
            # pickle_path = os.path.join(self.pickle_dir, f'{get_filename_from_path(doc_path)}_usechunks_{as_chunk}.pkl')
            
            # if os.path.exists(pickle_path):
            #     chunk_features, book_features = self.load_pickle(pickle_path)
            # else:
            chunk_features, book_features = self.get_doc_features_helper(doc_path, as_chunk, ngrams)
                #self.save_pickle(pickle_path, (chunk_features, book_features))


            all_chunk_features.extend(chunk_features)
            all_book_features.append(book_features)

        # Save book features only once (not when running with fulltext chunks)
        return all_chunk_features, all_book_features

    
    def get_corpus_features(self, as_chunk):
        cbfe = CorpusBasedFeatureExtractor(self.language, self.doc_paths, as_chunk, self.tokens_per_chunk)
        chunk_features, book_features = cbfe.get_all_features() ##############3
        # Save book features only once (not when running with fulltext chunks)
        return chunk_features, book_features ###########################
    
    def merge_features(self, doc_chunk_features, doc_book_features, doc_chunk_features_fulltext, corpus_chunk_features, corpus_book_features, corpus_chunk_features_fulltext):
        # Book features
        doc_book_features = pd.DataFrame(doc_book_features)
        doc_chunk_features = pd.DataFrame(doc_chunk_features)
        print(doc_chunk_features_fulltext)
        doc_chunk_features_fulltext = pd.DataFrame(doc_chunk_features_fulltext)

        print('doc_book_features: ', doc_book_features.shape, 'doc_chunk_features_fulltext: ', doc_chunk_features_fulltext.shape, 'corpus_book_features: ', corpus_book_features.shape, 'corpus_chunk_features_fulltext: ', corpus_chunk_features_fulltext.shape)

        book_df = doc_book_features\
                    .merge(right=doc_chunk_features_fulltext, on='file_name', how='outer', validate='one_to_one')\
                    .merge(right=corpus_book_features, on='file_name', validate='one_to_one')\
                    .merge(right=corpus_chunk_features_fulltext, on='file_name', validate='one_to_one')
        book_df.columns = [col + '_full' if col != 'file_name' else col for col in book_df.columns]

        # Chunk features
        chunk_df = doc_chunk_features.merge(right=corpus_chunk_features, on='file_name', how='outer', validate='one_to_one')
        # Remove chunk id from file_name
        chunk_df['file_name'] = chunk_df['file_name'].str.split('_').str[:4].str.join('_')
        chunk_df.columns = [col + '_chunk' if col != 'file_name' else col for col in chunk_df.columns]

        # Combine book features and averages of chunksaveraged chunk features
        # baac: book and averaged chunk
        baac_df = book_df.merge(chunk_df.groupby('file_name').mean().reset_index(drop=False), on='file_name', validate='one_to_many')
        # cacb: chunk and copied book
        cacb_df = chunk_df.merge(right=book_df, on='file_name', how='outer', validate='many_to_one')
        print(book_df.shape, chunk_df.shape, baac_df.shape, cacb_df.shape)

        dfs = {'book': book_df, 'baac': baac_df, 'chunk': chunk_df, 'cacb': cacb_df}
        for file_name, df in dfs.items():
            print(file_name)
            self.save_data(data=df, file_name=file_name)
    
    def run(self):
        gstart = time.time()

        start = time.time()
        path = os.path.join(self.pickle_dir, 'doc_chunk.pkl')
        if os.path.exists(path):
            doc_chunk_features, doc_book_features = self.load_pickle(path)
        else:
            doc_chunk_features, doc_book_features = self.get_doc_features(as_chunk=True)
            self.save_pickle(path, (doc_chunk_features, doc_book_features))
        print(f'Doc features: {time.time()-start}s.')

        start = time.time()
        path = os.path.join(self.pickle_dir, 'doc_full.pkl')
        if os.path.exists(path):
            doc_chunk_features_fulltext = self.load_pickle(path)[0]
            print('type doc_chunk_features_fulltext', type(doc_chunk_features_fulltext))
        else:
            # Recalculate the chunk features for the whole book, which is treated as one chunk
            doc_chunk_features_fulltext, _ = self.get_doc_features(as_chunk=False)
            self.save_pickle(path, (doc_chunk_features_fulltext))
        print(f'Doc full features: {time.time()-start}s.')


        print('STarting corpus features----------------------------\n')

        start = time.time()
        path = os.path.join(self.pickle_dir, 'corpus_chunk.pkl')
        if os.path.exists(path):
            corpus_chunk_features, corpus_book_features = self.load_pickle(path)
        else:
            corpus_chunk_features, corpus_book_features = self.get_corpus_features(as_chunk=True)
            self.save_pickle(path, (corpus_chunk_features, corpus_book_features))
        print(f'Corpus features: {time.time()-start}s.')
    

        start = time.time()
        path = os.path.join(self.pickle_dir, 'corpus_full.pkl')
        if os.path.exists(path):
            corpus_chunk_features_fulltext = self.load_pickle(path)[0]
        else:
            # Recalculate the chunk features for the whole book, which is considered as one chunk
            corpus_chunk_features_fulltext, _ = self.get_corpus_features(as_chunk=False)
            self.save_pickle(path, (corpus_chunk_features_fulltext))   
        print(f'Corpus full features: {time.time()-start}s.')   
        
        self.merge_features(
            doc_chunk_features,
            doc_book_features,
            doc_chunk_features_fulltext,
            corpus_chunk_features,
            corpus_book_features,
            corpus_chunk_features_fulltext
        )

        runtime = time.time() - gstart
        print('Runtime for all texts:', runtime)
        # with open('runtime_tracker.txt', 'a') as f:
        #     f.write('\n2, multiprocessing\n')
        #     f.write(f'nr_textruntime\n')
        #     f.write(f'{round(runtime, 2)}\n')

        

for language in ['eng']:
    fpath = '/home/annina/scripts/great_unread_nlp/data/features'
    ppath = '/home/annina/scripts/great_unread_nlp/data/pickle'
    if os.path.exists(ppath):
        shutil.rmtree(ppath)
    
    fe = FeatureProcessor(language).run()

# assert that nr of chunk features = nr of chunk in tokenizedwords







# %%
