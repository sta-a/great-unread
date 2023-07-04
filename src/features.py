# %%

%load_ext autoreload
%autoreload 2

import argparse
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
from feature_extraction.process_rawtext import Tokenizer
from feature_extraction.process_d2v import D2vProcessor
import sys
sys.path.append("..")
from utils import get_doc_paths, compare_directories, DataHandler, get_bookname

class FeatureProcessor(DataHandler):
    def __init__(self, language):
        self.output_dir = os.path.join(f'features', language)
        super().__init__(language, self.output_dir)
        self.raw_docs_dir = os.path.join(self.data_dir, 'raw_docs', self.language)
        self.doc_paths = get_doc_paths(self.raw_docs_dir)[:None] #############################
        self.dvs_chunks = D2vProcessor(self.language, output_dir=None, data_dir=self.data_dir).load_data(mode='chunk_features')
        self.dvs_doc = D2vProcessor(self.language, output_dir=None, data_dir=self.data_dir).load_data(mode='doc_tags')

    def tokenize_all_texts(self, remove_files=True):##############################
        if remove_files:
            chars_path = f'/home/annina/scripts/great_unread_nlp/src/special_chars_{self.language}.txt'
            if os.path.exists(chars_path):
                os.remove(chars_path)
            annotation_path = f'/home/annina/scripts/great_unread_nlp/src/annotation_words_{self.language}.txt'
            if os.path.exists(annotation_path):
                os.remove(annotation_path)
            for doc_path in self.doc_paths:
                tokenized_words_path = doc_path.replace('/raw_docs', '/tokenized_words')
                if os.path.exists(tokenized_words_path):
                    os.remove(tokenized_words_path)

        _ = Tokenizer(self.language, self.doc_paths, self.tokens_per_chunk).tokenize_all_texts()
        compare_directories(self.raw_docs_dir, self.raw_docs_dir.replace('/raw_docs', '/tokenized_words'))

    # def load_tokenized_words(self):
    #     chunks = Tokenizer(self.language, self.doc_paths, self.tokens_per_chunk).get_tokenized_words(self.doc_paths[0], remove_punct=False, lower=False, as_chunk=True)
    #     for i in chunks[:8]:
    #         print(i)
    #     no_chunks = Tokenizer(self.language, self.doc_paths, self.tokens_per_chunk).get_tokenized_words(self.doc_paths[0], remove_punct=False, lower=False, as_chunk=False)
    #     print(no_chunks[:100])


    def get_doc_features_helper(self, doc_path, use_chunks):
        if use_chunks:
            dvs = self.dvs_chunks
        else: 
            dvs = self.dvs_doc
        fe = DocBasedFeatureExtractor(self.language, doc_path, dvs, use_chunks)
        chunk_features, book_features = fe.get_all_features()
        return chunk_features, book_features
    

    def get_doc_features(self, use_chunks):
        all_chunk_features = []
        all_book_features = []

        for doc_path in self.doc_paths:
            pickled_path = doc_path.replace('/raw_docs', '/pickle') + f'_usechunks_{use_chunks}.pkl'
            pickled_dir = os.path.dirname(pickled_path)
            print(pickled_dir, pickled_path)
            os.makedirs(pickled_dir, exist_ok=True) 
            if os.path.exists(pickled_path):
                with open(pickled_path, 'rb') as f:
                    chunk_features, book_features = pickle.load(f)
                print(book_features)
            else:
                chunk_features, book_features = self.get_doc_features_helper(doc_path, use_chunks)
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
    
    def get_corpus_features(self, use_chunks):
        if use_chunks:
            dvs = self.dvs_chunks
        else: 
            dvs = self.dvs_doc
        cbfe = CorpusBasedFeatureExtractor(self.language, self.doc_paths, dvs, use_chunks)
        chunk_features, book_features = cbfe.get_all_features()
        # Save book features only once (not when running with fulltext chunks)
        return chunk_features, book_features
    
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
        return dfs
    
    def create_all_data(self):
        start = time.time()

        # self.load_tokenized_words()

        # Tokenize texts
        self.tokenize_all_texts()

        # # Doc-based features
        # doc_chunk_features, doc_book_features = self.get_doc_features(use_chunks=True)
        # # Recalculate the chunk features for the whole book, which is treated as one chunk
        # doc_chunk_features_fulltext, _ = self.get_doc_features(use_chunks=False)
        
        # # Corpus-based features
        # corpus_chunk_features, corpus_book_features = self.get_corpus_features(use_chunks=True)
        # # Recalculate the chunk features for the whole book, which is considered as one chunk
        # corpus_chunk_features_fulltext, _ = self.get_corpus_features(use_chunks=False)

        # dfs = self.merge_features(
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

if __name__ == '__main__':
    from_commandline = False
    if from_commandline:
        parser = argparse.ArgumentParser()
        parser.add_argument('--language', default='eng')
        parser.add_argument('--data_dir', default='../data')
        args = parser.parse_args()
        language = args.language
        data_dir = args.data_dir
    else:
        # Don't use defaults because VSC Python interactive mode can't handle command line arguments
        language = 'eng'
        data_dir = '../data/'

    # Select number of texts to work with
    tokens_per_chunk = 500

    # ngram_counts_path = os.path.join(Path(str(doc_paths[0].replace('raw_docs', 'ngram_counts'))).parent, 'ngram_counts.pkl')
    # if os.path.exists(ngram_counts_path):
    #     os.remove(ngram_counts_path)
    fe = FeatureProcessor(language).create_all_data()

# assert that nr of chunk features = nr of chunk in tokenizedwords

# %%
