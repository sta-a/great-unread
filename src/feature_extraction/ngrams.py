# %%
# %load_ext autoreload
# %autoreload 2

import pandas as pd
import joblib
import pickle
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from .process_rawtext import ChunkHandler
import sys
sys.path.append("..")
from utils import DataHandler, get_filename_from_path, get_files_in_dir
import os
import numpy as np
import time
import re


class NgramCounter(DataHandler):
    def  __init__(self, language):
        super().__init__(language, output_dir='ngram_counts', data_type='pkl')
        self.nr_chunks_check = {}
        self.nr_chunknames_check = {}
        self.unigrams = ['unigram', 'bigram', 'trigram']
        self.sizes = ['full', 'chunk']
        self.modes = [(item1, item2) for item1 in self.unigrams for item2 in self.sizes]
        self.data_dict = None
        self.ch = ChunkHandler(self.language, self.tokens_per_chunk)


    def load_chunks(self, mode):
        # Generator for loading tokenized chunks
        ntype, size = mode
        if size == 'chunk':
            as_chunk = True
        else:
            as_chunk = False

        if ntype == 'unigram':
            as_sent = False
        else: 
            as_sent = True

        print('ntype: ', ntype, 'size: ', size)
        for doc_path in self.doc_paths:
            bookname = get_filename_from_path(doc_path)
            chunks = self.ch.load_data(file_name=bookname, remove_punct=False, lower=False, as_chunk=as_chunk, as_sent=as_sent)
            self.nr_chunks_check[bookname] = len(chunks)

            if as_sent:
                new_chunks = []
                for chunk in chunks:
                    sentences = ['<BOS> ' + sent + ' <EOS>' for sent in chunk]
                    chunk = ' '.join(sentences)   
                    new_chunks.append(chunk) # Create simple list of strings, strings represent chunks  
                chunks = new_chunks

            for chunk in chunks:
                yield chunk
                
    def get_chunk_names(self, size):
        def count_lines(file_path):
            with open(file_path, 'r') as f:
                return sum(1 for _ in f)
            
        chunk_names = []
        file_list = get_files_in_dir(self.ch.output_dir)

        if size == 'chunk':
            for file in file_list:
                file_path = os.path.join(self.ch.output_dir, file)
                nr_lines = count_lines(file_path)
                self.nr_chunknames_check[get_filename_from_path(file_path)] = nr_lines
                chunk_names.extend([f"{get_filename_from_path(file_path)}_{i}" for i in range(0, nr_lines)])
        else:
            chunk_names = [get_filename_from_path(dp) for dp in self.doc_paths]
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

        file_names = self.get_chunk_names(size)
        dtm = cv.fit_transform(self.load_chunks(mode))
        words = cv.get_feature_names_out()
        words = ['-'.join(name.split()) for name in words] # for bi- and trigrams, join words
        # Create a dictionary to store everything
        self.data_dict = {
            'dtm': dtm,
            'words': words,
            'file_names': file_names
        }

        if size == 'chunk':
            print(self.nr_chunknames_check, self.nr_chunks_check)
            assert self.nr_chunknames_check == self.nr_chunks_check, f'nr_chunknames_check and nr_chunks_check are not equal'
            print(len(list(self.nr_chunknames_check.keys())), len(list(self.nr_chunks_check.keys())))
            a=list(self.nr_chunknames_check.keys())
            b=list(self.nr_chunks_check.keys())
            for i, j in zip(a,b):
                print(i, j)
            assert list(self.nr_chunknames_check.keys()) == list(self.nr_chunks_check.keys()), f'keys are not equal'
        # Save the dictionary to a file
        self.save_data(data=self.data_dict, ntype=mode[0], size=mode[1])

    def load_data(self, load=True, file_name=None, **kwargs):  
        self.data_dict = super().load_data(load=load, file_name=file_name, **kwargs)

    def load_values_for_chunk(self, file_name):
        if self.data_dict is None:
            raise ValueError(f'Load data before loading values for a specific file.')
        
        # Find the index of the selected label in the labels list
        selected_index = self.data_dict['file_names'].index(file_name)

        # Access the term frequency values for the specific document
        term_frequencies_for_selected_document = self.data_dict['dtm'][selected_index].toarray()
        print('term_frequencies_for_selected_document', type(term_frequencies_for_selected_document), term_frequencies_for_selected_document)

        # Convert the term frequency values to a dictionary for better understanding
        file_ngram_counts = {self.data_dict['words'][i]: term_frequencies_for_selected_document[0][i] for i in range(len(self.data_dict['words']))}
        return file_ngram_counts
    
    def check_data(self):
        dc = self.DataChecker(self.language, ngrams_dir=self.output_dir)
        dc.check_filenames()
        df = dc.get_total_unigram_freq()
        dc.plot_zipfs_law(df)


    class DataChecker(DataHandler):
        '''
        Class for checking chunking
        '''
        def __init__(self, language, ngrams_dir):
            super().__init__(language, output_dir='text_statistics', data_type='svg')
            self.ngrams_dir = ngrams_dir
            self.nc = NgramCounter(language=self.language)

        def check_filenames(self):
            ch = ChunkHandler(self.language, self.tokens_per_chunk)
            nr_chunks_per_doc, total_nr_chunks = ch.DataChecker(self.language, ch.output_dir).count_chunks_per_doc()
            nr_texts = len(nr_chunks_per_doc)
            for unigram in self.unigrams:
                for size in self.sizes:
                    self.nc.load_data(file_name=f'{unigram}_{size}')

                    fn = self.nc.data_dict['file_names']
                    nr_fn = len(fn)
                    print(f'nr texts: {nr_texts}, nr filenames: {nr_fn}')

                    if size == 'full':
                        assert nr_fn ==   nr_texts
                    else:
                        assert nr_fn == total_nr_chunks


        def get_total_unigram_freq(self):
            self.nc.load_data(mode='unigram_full')
            words = self.nc.data_dict['words']

            dtm = self.nc.data_dict['dtm']

            # Ensure dtm is a dense NumPy array
            dtm_dense = dtm.toarray()          
            # Calculate the sum along axis 0
            dtm_sum = np.sum(dtm_dense, axis=0)

            data = {'word': words, 'count': dtm_sum}
            df = pd.DataFrame(data)
            self.save_data(data=df, file_name='unigram_counts.csv', pandas_index=False)
            return df
        
        def plot_zipfs_law(self, df):
            df = df.sort_values(by='count', ascending=False)
            ranks = np.arange(1, df.shape[0] + 1)


            # Calculate expected frequencies based on Zipf's Law
            k = df['count'].iloc[0]  # Take the count of the top-ranked word as k
            expected_freq = k / ranks

            # Plot Zipf's Law on a log-log scale
            plt.figure(figsize=(10, 6))
            plt.loglog(ranks, expected_freq, linestyle='-', color='r', label="Zipf's Law")
            plt.loglog(ranks, df['count'], marker='.', linestyle='None', color='b', label='Counted')
            plt.xlabel('Rank')
            plt.ylabel('Frequency')
            plt.title("Zipf's Law (log-log scale)")
            plt.legend()
            plt.grid(True)

            for i in range(4):
                plt.annotate(f'{df.iloc[i]["word"]}', (ranks[i], df.iloc[i]["count"]), textcoords='offset points', xytext=(0, 5))

            count_greater_than_1 = len(df[df['count'] > 1])

            plt.text(0.99, 0.85, f'Words with count > 1: {count_greater_than_1:,}\nNr. unique words: {df.shape[0]:,}', transform=plt.gca().transAxes, 
                     va='top', ha='right', bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))                    
            # # Annotate the plot with words at every power of 10 frequency
            # for power in range(1, int(np.log10(df['count'].max())) + 1):
            #     freq_threshold = 10 ** power
            #     closest_index = np.argmin(np.abs(df['count'] - freq_threshold))
            #     word_to_annotate = df.iloc[closest_index]['word']
            #     plt.annotate(f'{word_to_annotate}', (ranks[closest_index], df['count'].iloc[closest_index]), textcoords='offset points', xytext=(0, 5))


            self.save_data(file_name='zipfs-law', data=plt, data_type='png')


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
