# %%
%load_ext autoreload
%autoreload 2

import os
import pickle
import numpy as np
import pandas as pd
import time
from itertools import repeat
from multiprocessing import Pool, cpu_count
from pathlib import Path
from feature_extraction.doc_based_feature_extractor import DocBasedFeatureExtractor
from feature_extraction.corpus_based_feature_extractor import CorpusBasedFeatureExtractor
from feature_extraction.embeddings import SbertProcessor, D2vProcessor

import sys
sys.path.append("..")
from utils import DataHandler


class FeatureProcessor(DataHandler):
    def __init__(self, language):
        super().__init__(language, 'features')
        self.doc_paths = self.doc_paths[:None]

    def get_doc_features_helper(self, doc_path, as_chunk):
        fe = DocBasedFeatureExtractor(self.language, doc_path, as_chunk)
        chunk_features, book_features = fe.get_all_features()
        return chunk_features, book_features
    

    def get_doc_features(self, as_chunk):
        all_chunk_features = []
        all_book_features = []

        for doc_path in self.doc_paths:
            pickled_path = doc_path.replace('/text_raw', '/pickle') + f'_usechunks_{as_chunk}.pkl'
            pickled_dir = os.path.dirname(pickled_path)
            print(pickled_dir, pickled_path)
            os.makedirs(pickled_dir, exist_ok=True) 
            if os.path.exists(pickled_path):
                with open(pickled_path, 'rb') as f:
                    chunk_features, book_features = pickle.load(f)
                print(book_features)
            else:
                chunk_features, book_features = self.get_doc_features_helper(doc_path, as_chunk)
                with open(pickled_path, 'wb') as f:
                    pickle.dump((chunk_features, book_features), f)
                    self.logger.info(f'')

            all_chunk_features.extend(chunk_features)
            all_book_features.append(book_features)

        print(len(all_chunk_features), len(all_book_features)) ##################################
        # Save book features only once (not when running with fulltext chunks)
        return all_chunk_features, all_book_features
       
    # def get_doc_features(self):
    #     all_chunk_features = []
    #     all_book_features = []

    #     nr_processes = max(cpu_count() - 2, 1)
    #     with Pool(processes=nr_processes) as pool:
    #         # res = pool.map(self.get_doc_features_helper, self.doc_paths) #ChatGPT
    #         # res = pool.starmap(self.get_doc_features_helper, zip(self.doc_paths))
    #         res = pool.map(self.get_doc_features_helper, self.doc_paths)
    #         for doc_features in res:
    #             all_chunk_features.extend(doc_features[0])
    #             all_book_features.append(doc_features[1])

    #     print(len(all_chunk_features), len(all_book_features)) ##################################
    #     # Save book features only once (not when running with fulltext chunks)
    #     return all_chunk_features, all_book_features
    
    def get_corpus_features(self, as_chunk):
        cbfe = CorpusBasedFeatureExtractor(self.language, self.doc_paths, as_chunk)
        # chunk_features, book_features = cbfe.get_all_features() ##############3
        print('cbfe get all features not called§')
        # Save book features only once (not when running with fulltext chunks)
        # return chunk_features, book_features ###########################
    
    def merge_features(self, doc_chunk_features, doc_book_features, doc_chunk_features_fulltext, corpus_chunk_features, corpus_book_features, corpus_chunk_features_fulltext):
        # Book features
        doc_book_features = pd.DataFrame(doc_book_features)
        doc_chunk_features_fulltext = pd.DataFrame(doc_chunk_features_fulltext)

        print('doc_book_features: ', doc_book_features.shape, 'doc_chunk_features_fulltext: ', doc_chunk_features_fulltext.shape, 'corpus_book_features: ', corpus_book_features.shape, 'corpus_chunk_features_fulltext: ', corpus_chunk_features_fulltext.shape)

        book_df = doc_book_features\
                    .merge(right=doc_chunk_features_fulltext, on='file_name', how='outer', validate='one_to_one')\
                    .merge(right=corpus_book_features, on='file_name', validate='one_to_one')\
                    .merge(right=corpus_chunk_features_fulltext, on='file_name', validate='one_to_one')
        book_df.columns = [col + '_fulltext' if col != 'file_name' else col for col in book_df.columns]

        # Chunk features
        doc_chunk_features = pd.DataFrame(doc_chunk_features)
        chunk_df = doc_chunk_features.merge(right=corpus_chunk_features, on='file_name', how='outer', validate='one_to_one')
        # Remove chunk id from file_name
        chunk_df['file_name'] = chunk_df['file_name'].str.split('_').str[:4].str.join('_')
        chunk_df.columns = [col + '_chunk' if col != 'file_name' else col for col in chunk_df.columns]

        # Combine book features and averages of chunksaveraged chunk features
        # baac = book and averaged chunk
        baac_df = book_df.merge(chunk_df.groupby('file_name').mean().reset_index(drop=False), on='file_name', validate='one_to_many')
        # cacb = chunk and copied book
        cacb_df = chunk_df.merge(right=book_df, on='file_name', how='outer', validate='many_to_one')
        print(book_df.shape, chunk_df.shape, baac_df.shape, cacb_df.shape)

        dfs = {'book': book_df, 'baac': baac_df, 'chunk': chunk_df, 'cacb': cacb_df}
        for file_name, df in dfs.items():
            print(file_name)
            self.save_data(data=df, file_name=file_name)
    
    def run(self):
        start = time.time()

        doc_chunk_features, doc_book_features = self.get_doc_features(as_chunk=True)
        # Recalculate the chunk features for the whole book, which is treated as one chunk
        # doc_chunk_features_fulltext, _ = self.get_doc_features(as_chunk=False)
        
        # corpus_chunk_features, corpus_book_features = self.get_corpus_features(as_chunk=True)
        # # Recalculate the chunk features for the whole book, which is considered as one chunk
        # corpus_chunk_features_fulltext, _ = self.get_corpus_features(as_chunk=False)

        # self.merge_features(
        #     doc_chunk_features,
        #     doc_book_features,
        #     doc_chunk_features_fulltext,
        #     corpus_chunk_features,
        #     corpus_book_features,
        #     corpus_chunk_features_fulltext
        # )

        runtime = time.time() - start
        print('Runtime for all texts:', runtime)
        with open('runtime_tracker.txt', 'a') as f:
            f.write('\n2, multiprocessing\n')
            f.write(f'nr_textruntime\n')
            f.write(f'{round(runtime, 2)}\n')

for language in ['eng']:
    fe = FeatureProcessor(language).run()

# assert that nr of chunk features = nr of chunk in tokenizedwords

# %%
