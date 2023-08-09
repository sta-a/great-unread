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
from feature_extraction.embeddings import D2vProcessor, SbertProcessor
from feature_extraction.ngrams import NgramCounter, MfwExtractor
import sys
sys.path.append("..")
from utils import get_doc_paths, check_equal_files, DataHandler, get_bookname

class FeaturePreparer(DataHandler):
    def __init__(self, language):
        super().__init__(language, 'features')
        self.text_raw_dir = os.path.join(self.data_dir, 'text_raw', self.language)
        self.doc_paths = get_doc_paths(self.text_raw_dir)[:None] 
    '''
    Extract basic features that take a while to process before using more detailed processing.
    This is not necessary, since all data preparation steps are also called from the detailed processing, but is practical.
    '''
    def load_text_tokenized(self):
        chunks = Tokenizer(self.language, self.doc_paths, self.tokens_per_chunk).create_data(self.doc_paths[0], remove_punct=False, lower=False, as_chunk=True)
        no_chunks = Tokenizer(self.language, self.doc_paths, self.tokens_per_chunk).create_data(self.doc_paths[0], remove_punct=False, lower=False, as_chunk=False)

    def tokenizer(self, remove_files=False):
        if remove_files:
            chars_path = f'/home/annina/scripts/great_unread_nlp/src/special_chars_{self.language}.txt'
            if os.path.exists(chars_path):
                os.remove(chars_path)
            annotation_path = f'/home/annina/scripts/great_unread_nlp/src/annotation_words_{self.language}.txt'
            if os.path.exists(annotation_path):
                os.remove(annotation_path)
            for doc_path in self.doc_paths:
                text_tokenized_path = doc_path.replace('text_raw', 'text_tokenized')
                if os.path.exists(text_tokenized_path):
                    os.remove(text_tokenized_path)

        t = Tokenizer(self.language, self.doc_paths, self.tokens_per_chunk)
        # t.create_all_data()
        t.check_data()

    def ngramcounter(self):
        c = NgramCounter(self.language)
        c.create_all_data()
        c.get_total_unigram_freq()
        # c.create_all_data()
        # c.load_data(file_name='unigram_chunk.pkl')
        # print(c.data_dict['words'][1500:1600])
        # # df = c.load_data('trigram_chunks.pkl')
        #file_ngram_counts = c.load_values_for_chunk(file_name='Ainsworth_William-Harrison_Rookwood_1834_0')
        # for k, v in file_ngram_counts.items():
        #     print(k, v)

    def mfwextractor(self):
        m = MfwExtractor(self.language)
        m.create_all_data()

    def sbert(self):
        s = SbertProcessor(self.language)
        s.create_all_data()

    def run(self):
        start = time.time()
        # self.load_text_tokenized()
        # self.tokenizer()
        self.ngramcounter()
        # self.mfwextractor()
        # self.sbert()
        print('Time: ', time.time()-start)


for language in ['eng']:#, 'ger']: 
    fp = FeaturePreparer(language)
    fp.run()

# %%


class FeatureProcessor(DataHandler):
    def __init__(self, language):
        super().__init__(language, 'features')
        self.text_raw_dir = os.path.join(self.data_dir, 'text_raw', self.language)
        self.doc_paths = get_doc_paths(self.text_raw_dir)[:None] #############################

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
            pickled_path = doc_path.replace('/text_raw', '/pickle') + f'_usechunks_{use_chunks}.pkl'
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
        return dfs
    
    def run(self):
        start = time.time()

        # Create d2v embeddings
        _ = D2vProcessor(self.language).load_all_data()

        # Load document vectors after texts have been tokenized
        # This order is not strictly necessary but more convenient
        self.dvs_chunks = D2vProcessor(self.language).load_data(mode='chunk_features')
        self.dvs_doc = D2vProcessor(self.language).load_data(mode='doc_tags')

        # # Doc-based features
        # doc_chunk_features, doc_book_features = self.get_doc_features(use_chunks=True)
        # # Recalculate the chunk features for the whole book, which is treated as one chunk
        # doc_chunk_features_fulltext, _ = self.get_doc_features(use_chunks=False)
        
        # # Corpus-based features
        #corpus_chunk_features, corpus_book_features = 
        # self.get_corpus_features(use_chunks=True)
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
        language = 'ger'

    fe = FeatureProcessor(language).run()

# assert that nr of chunk features = nr of chunk in tokenizedwords

# %%
