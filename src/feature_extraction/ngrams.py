# %%
# %load_ext autoreload
# %autoreload 2

import pandas as pd
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from .process_rawtext import TextLoader
import sys
sys.path.append("..")
from utils import DataHandler, get_filename_from_path
import os
import numpy as np
import time
import re


class NgramCounter(DataHandler):
    def  __init__(self, language):
        super().__init__(language, output_dir='ngram_counts', data_type='pkl')
        self.text_tokenized_dir = os.path.join(self.data_dir, 'text_tokenized', self.language)
        self.nr_chunks_check = {}
        self.nr_chunknames_check = {}
        self.unigrams = ['unigram', 'bigram', 'trigram']
        self.sizes = ['full', 'chunk']
        self.modes = [(item1, item2) for item1 in self.unigrams for item2 in self.sizes]
        self.terminating_chars = r'\. | \: | \; | \? | \! | \) | \] | \...'
        self.data_dict = None


    def load_chunks(self, mode):
        # Generator for loading tokenized chunks
        ntype, size = mode
        print('ntype', ntype, 'size', size)
        for doc_path in self.doc_paths[:5]:
            if size == 'chunk':
                chunks = TextLoader(self.language, self.doc_paths, self.tokens_per_chunk).load_data(doc_path, remove_punct=False, lower=False, as_chunk=True)
            else:
                chunks = [TextLoader(self.language, self.doc_paths, self.tokens_per_chunk).load_data(doc_path, remove_punct=False, lower=False, as_chunk=False)]
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
        file_list = [os.path.join(self.text_tokenized_dir, os.path.basename(dp)) for dp in self.doc_paths]

        if size == 'chunk':
            for file_path in file_list:
                nr_lines = count_lines(file_path)
                self.nr_chunknames_check[get_filename_from_path(file_path)] = nr_lines
                chunk_names.extend([f"{os.path.splitext(get_filename_from_path(file_path))[0]}_{i}" for i in range(0, nr_lines)])
        else:
            chunk_names = [os.path.splitext(get_filename_from_path(dp))[0] for dp in self.doc_paths]
        return chunk_names

    def create_data(self, mode):
        ntype, size = mode
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

        dtm = cv.fit_transform(self.load_chunks(mode))
        words = cv.get_feature_names_out()
        words = ['-'.join(name.split()) for name in words] # for bi- and trigrams, join words
        file_names = self.get_chunk_names(size)
        # Create a dictionary to store everything
        self.data_dict = {
            'dtm': dtm,
            'words': words,
            'file_names': file_names
        }

        # if size == 'chunk':
        #     assert self.nr_chunknames_check == self.nr_chunks_check
        #     assert list(self.nr_chunknames_check.keys()) == list(self.nr_chunks_check.keys())
        # # Save the dictionary to a file
        self.save_data(data=self.data_dict, ntype=mode[0], size=mode[1])


    def to_df(self, dtm, words, file_names):
        df = pd.DataFrame(dtm.toarray(), columns=words, index=file_names)
        # Sort columns by the absolute frequencies of the words in the corpus
        # df = df[df.sum().sort_values(ascending=False).index]
        return df
    
    def load_values_for_chunk(self, file_name):
        if self.data_dict is None:
            raise ValueError(f'Load data before loading values for a specific file.')
        
        # Find the index of the selected label in the labels list
        selected_index = self.data_dict['file_names'].index(file_name)

        # Access the term frequency values for the specific document
        term_frequencies_for_selected_document = self.data_dict['dtm'][selected_index].toarray()
        print('term_frequencies_for_selected_document', type(term_frequencies_for_selected_document), term_frequencies_for_selected_document)

        print(self.data_dict['words'])
        # Convert the term frequency values to a dictionary for better understanding
        file_ngram_counts = {self.data_dict['words'][i]: term_frequencies_for_selected_document[0][i] for i in range(len(self.data_dict['words']))}
        return file_ngram_counts
    
    def get_total_unigram_freq(self):
        self.load_data(mode='unigram_full')
        word_count_dict = {}
        words = self.data_dict['words']
        for idx, word in enumerate(words):
            word_count_dict[word] = self.data_dict['dtm'][:, idx].sum()

        # Print the words and their counts
        for word, count in word_count_dict.items():
            print(f"{word}: {count}")

        self.save_data(data=word_count_dict, file_name='unigram_counts.txt', data_type='dict')


    def load_data(self, load=True, file_name=None, **kwargs):  
        self.data_dict = super().load_data(load=load, file_name=file_name, **kwargs)


    # def load_all_data(self): ##############################
    #     print('load all data')
    #     all_data = {}
    #     for mode in self.modes:
    #         data  = self.load_data(mode=mode)

class MfwExtractor(DataHandler):
    '''
    Calculate MFW tables from word count matrix.
    '''
    def  __init__(self, language):
        super().__init__(language, output_dir='ngram_counts', data_type='csv')
        self.df = self.prepare_df()
        self.modes = [500, 1000, 2000, 5000]

    def prepare_df(self):
        df = NgramCounter(self.language).load_data(file_name='unigram_full.pkl')
        # Calculate relative frequencies
        df = df.divide(df.sum(axis=1), axis=0)
        # Sort according to relative frequencies
        df = df[df.sum().sort_values(ascending=False).index]
        return df

    def create_data(self, **kwargs):
        mfw = kwargs['mode']
        df = self.df.iloc[:, :mfw]
        print(df.sum())
        self.save_data(data=df, file_name=None, **kwargs)

    def create_filename(self, **kwargs):
        file_name = super().create_filename(**kwargs, file_string='mfw')
        print(file_name)
        return file_name
    
# %%
