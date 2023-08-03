# %%
# %load_ext autoreload
# %autoreload 2

import pandas as pd
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from process_rawtext import Tokenizer
import sys
sys.path.append("..")
from utils import get_doc_paths, DataHandler, get_filename_from_path
import os
import numpy as np
import time
import re

class CVobject(DataHandler):
    def  __init__(self, language):
        super().__init__(language, output_dir='ngram_counts', data_type='pkl')
        self.raw_docs_dir = os.path.join(self.data_dir, 'raw_docs', self.language)
        self.tokenized_words_dir = os.path.join(self.data_dir, 'tokenized_words', self.language)
        self.doc_paths = sorted(get_doc_paths(self.raw_docs_dir))
        self.nr_chunks_check = {}
        self.nr_chunknames_check = {}
        self.modes = [('unigram', 'full'), ('bigram', 'full'), ('bigram', 'chunk'), ('trigram', 'full'), ('trigram', 'chunk')]
        self.terminating_chars = r'\. | \: | \; | \? | \! | \) | \] | \...'


    def load_chunks(self, ntype, size):
        # Generator for loading tokenized chunks
        for doc_path in self.doc_paths:
            if size == 'chunks':
                chunks = Tokenizer(self.language, self.doc_paths, self.tokens_per_chunk).get_tokenized_words(doc_path, remove_punct=False, lower=False, as_chunk=True)
            else:
                chunks = [Tokenizer(self.language, self.doc_paths, self.tokens_per_chunk).get_tokenized_words(doc_path, remove_punct=False, lower=False, as_chunk=False)]
            self.nr_chunks_check[get_filename_from_path(doc_path)] = len(chunks)

            for chunk in chunks:              
                if ntype != 'unigram':
                    sentences = re.split(self.terminating_chars, chunk)
                    sentences = ['<BOS> ' + sent + ' <EOS>' for sent in sentences]
                    chunk = ' '.join(sentences)           
                yield chunk

    def get_chunk_names(self, size):
        def count_lines(file_path):
            with open(file_path, 'r') as f:
                return sum(1 for _ in f)
            
        chunk_names = []
        file_list = [os.path.join(self.tokenized_words_dir, os.path.basename(dp)) for dp in self.doc_paths]

        if size == 'chunks':
            for file_path in file_list:
                nr_lines = count_lines(file_path)
                self.nr_chunknames_check[get_filename_from_path(file_path)] = nr_lines
                chunk_names.extend([f"{os.path.splitext(get_filename_from_path(file_path))[0]}_{i}" for i in range(0, nr_lines)])
        else:
            chunk_names = [os.path.splitext(get_filename_from_path(dp))[0] for dp in self.doc_paths]
        return chunk_names
    
    def counts_words(self, ntype, size):
        if ntype == 'unigram':
            ngram_range = (1, 1)
            # CV's default regex pattern only matches strings that are at least 2 chars long.
            # This pattern also matches strings that are only 1 char long (a, I...).
            cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=ngram_range, dtype=np.int32)
        else:
            if ntype == 'bigram':
                ngram_range = (2, 2)
            else:
                ngram_range = (3, 3)
            cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=ngram_range, dtype=np.int32, max_features=2000)

        df = cv.fit_transform(self.load_chunks(ntype, size))
        words = cv.get_feature_names_out()
        words = ['-'.join(name.split()) for name in words] # for bi- and trigrams, join words
        return df, words


    def to_df(self, cv, words, ntype, size):
        names = self.get_chunk_names(size)
        df = pd.DataFrame(cv.toarray(), columns=words, index=names)
        # Sort columns by the absolute frequencies of the words in the corpus
        df = df[df.sum().sort_values(ascending=False).index]
        self.save_data(data=df, ntype=ntype, size=size)

    def create_data(self, ntype, size):
        df, words = self.counts_words(ntype, size)

        # joblib.dump(df, 'count_vectorizer.pkl')
        self.to_df(df, words, ntype, size)
        if size == 'chunks':
            assert self.nr_chunknames_check == self.nr_chunks_check
            assert list(self.nr_chunknames_check.keys()) == list(self.nr_chunks_check.keys())

    def create_all_data(self):
        for ntype, size in self.modes:
            start = time.time()
            print(ntype, size)
            self.create_data(ntype, size)
            print(f'{time.time()-start}s to calculate {ntype} {size}.')


c = CVobject('eng')
c.create_all_data()


# %%
class MFW(DataHandler):
    '''
    Calculate MFW tables from word count matrix.
    '''
    def  __init__(self, language):
        super().__init__(language, output_dir='ngram_counts', data_type='pkl')
        self.df = self.prepare_df()
        self.modes = [500, 1000, 2000, 5000]

    def prepare_df(self):
        df = CVobject(self.language).load_data(file_name='unigram_full.pkl')
        # Calculate relative frequencies
        df = self.df.divide(self.df.sum(axis=1), axis=0)
        # Sort according to relative frequencies
        df = df[df.sum().sort_values(ascending=False).index]

    def create_data(self, **kwargs):
        mfw = kwargs['mode']
        mfw = self.df.iloc[:, :mfw]
        self.save_data(data=mfw, file_name=None, **kwargs) ######## mode statt kwargs

    def create_filename_base(self, **kwargs):
        data_type = self.get_custom_datatype(**kwargs)
        return f"mfw{str(kwargs['mode'])}.{data_type}"
    
    def create_all_data(self):
        for mode in self.modes:
            self.create_data(mode=mode)