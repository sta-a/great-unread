# %%
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
np.random.seed(6)
import time
import re


class NgramCounter(DataHandler):
    def  __init__(self, language):
        super().__init__(language, output_dir='ngram_counts', data_type='pkl', test=False)
        self.nr_chunknames_check = {}
        # self.unigrams = ['unigram', 'bigram', 'trigram']
        # self.sizes = ['full', 'chunk']
        self.unigrams = ['trigram'] ######################################
        self.sizes = ['full']
        self.modes = [(item1, item2) for item1 in self.unigrams for item2 in self.sizes]
        self.ch = ChunkHandler(self.language, self.tokens_per_chunk)
        self.chunk_names = []

    def get_chunknames(self, doc_path, nr_chunks, size):
        bookname = get_filename_from_path(doc_path)
        if size == 'chunk':
            for idx in range(0, nr_chunks):
                fn = f'{bookname}_{idx}'
                self.chunk_names.append(fn)
        else:
            self.chunk_names.append(get_filename_from_path(doc_path))


    def load_chunks(self, mode):
        self.chunk_names = []
        # Generator for loading tokenized chunks
        ntype, size = mode
        if size == 'chunk':
            as_chunk = True
        else:
            as_chunk = False

        print('ntype: ', ntype, 'size: ', size)
        for doc_path in self.doc_paths:
            bookname = get_filename_from_path(doc_path)
            chunks = self.ch.load_data(file_name=bookname, remove_punct=False, lower=False, as_chunk=as_chunk, as_sent=False)
            self.get_chunknames(doc_path, len(chunks), size)
            
            for chunk in chunks:
                yield chunk
                

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
                # min_df = 10 #7
            else:
                ngram_range = (3, 3)
                # min_df = 10 #8
            # cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=ngram_range, dtype=np.int32, max_features=2000)
            cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=ngram_range, dtype=np.int32, min_df=1)

        dtm = cv.fit_transform(self.load_chunks(mode))
        words = cv.get_feature_names_out()
        words = ['-'.join(name.split()) for name in words] # for bi- and trigrams, join words
        # Create a dictionary to store everything
        data_dict = {
            'dtm': dtm,
            'words': words,
            'file_names': self.chunk_names
        }

        # Save the dictionary to a file
        self.save_data(data=data_dict, ntype=mode[0], size=mode[1])


    def load_data(self, load=True, file_name=None, **kwargs):  
        data_dict = super().load_data(load=load, file_name=file_name, **kwargs)
        return data_dict


    def load_values_for_chunk(self, file_name, data_dict):
        if data_dict is None:
            raise ValueError(f'Load data before loading values for a specific file.')
        
        # Find the index of the selected label in the labels list
        idx = data_dict['file_names'].index(file_name)

        # Access the term frequency values for the specific document
        file_counts = data_dict['dtm'][idx].toarray()

        file_counts = {data_dict['words'][i]: file_counts[0][i] for i in range(len(data_dict['words']))}
        return file_counts
    

    def load_all_ngrams(self, as_chunk=None, size=None):
        if as_chunk is None and size is None:
            raise ValueError("Either as_chunk or size must have a value, but both are None")
        
        # Check if both arguments are not None
        if as_chunk is not None and size is not None:
            raise ValueError("Both as_chunk and size cannot have values at the same time")
    
        if as_chunk is not None:
            if as_chunk:
                size = 'chunk'
            else:
                size = 'full'
        
        unigrams = self.load_data(file_name=f'unigram_{size}')
        bigrams = self.load_data(file_name=f'bigram_{size}')
        trigrams = self.load_data(file_name=f'trigram_{size}')

        self.logger.info(f'Returning ngram data dicts.')
        return {'unigram': unigrams, 'bigram': bigrams, 'trigram': trigrams}
    

    def filter_dtm(self, data_dict, min_docs=None, min_percent=None, max_docs=None, max_percent=None):
        if min_docs is None and min_percent is None and max_docs is None and max_percent is None:
            raise ValueError('Specify at least one filtering criterion.')

        dtm = data_dict['dtm']
        words = data_dict['words']

        # Filter minimum
        if min_docs is not None and min_percent is not None:
            raise ValueError('Specify either the the minimum number or the minimum percent.')
        if min_percent is not None:
            min_docs = round(min_percent/100 * dtm.shape[0])
        if min_docs is not None:
            # Boolean mask dtm > 0: True if element is greater 0
            # Sum over True values -> count in how many docs the word occurs
            doc_frequency = np.array(np.sum(dtm > 0, axis=0))[0]
            min_columns = np.where(doc_frequency >= min_docs)[0]
            dtm = dtm[:, min_columns]
            words = [words[i] for i in min_columns]

        # Filter maximum
        if max_docs is not None and max_percent is not None:
            raise Exception('Specify either the the maximum number or the maximum percent.')
        if max_percent is not None:
            max_docs = round(max_percent/100 * dtm.shape[0])
        if max_docs is not None:
            doc_frequency = np.array(np.sum(dtm > 0, axis=0))[0] # Recalculate for reduced dtm
            max_columns = np.where(doc_frequency <= max_docs)[0]
            dtm = dtm[:, max_columns]
            words = [words[i] for i in max_columns]

        return dtm, words
        

    def check_data(self):
        dc = self.DataChecker(self.language, ngrams_dir=self.output_dir)
        dc.check_filenames()
        dc.check_rare_words()
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
            for unigram in self.nc.unigrams:
                for size in self.nc.sizes:
                    data_dict = self.nc.load_data(file_name=f'{unigram}_{size}')

                    fn = data_dict['file_names']
                    nr_fn = len(fn)
                    print(f'nr texts: {nr_texts}, nr filenames: {nr_fn}')

                    if size == 'full':
                        assert nr_fn ==   nr_texts
                    else:
                        assert nr_fn == total_nr_chunks


        def check_rare_words(self):
            '''
            Print some words that occur only once in the corpus and the file in which they occur.
            '''
            data_dict = self.nc.load_data(mode='unigram_chunk')
            words = data_dict['words']

            dtm = data_dict['dtm']

            # Calculate the sum along axis 0 (columns) directly on the sparse matrix
            dtm_sum = np.array(dtm.sum(axis=0))[0]

            # Randomly select a word with count 1
            unique_word_indices = np.where(dtm_sum == 1)[0]

            for i in range(0, 10):
                    # Find the indices where the unique word appears in the DTM
                    random_unique_word_index = np.random.choice(unique_word_indices)
                    
                    # Get the indices of documents where this word appears directly from the sparse matrix
                    document_indices = dtm[:, random_unique_word_index].nonzero()[0]
                    
                    # Retrieve the file name associated with the first document where the word appears
                    idx = data_dict['file_names'][int(document_indices[0])]
                    
                    # Print the randomly selected word and the indices of documents where it appears
                    print(f"Randomly selected word with count 1: '{words[random_unique_word_index]}'")
                    print(f"Indices of documents where this word appears: {idx}")


        def get_total_unigram_freq(self):
            data_dict = self.nc.load_data(mode='unigram_full')
            words = data_dict['words']

            dtm = data_dict['dtm']
   
            # Calculate the sum along axis 0
            dtm_sum = list(np.array(dtm.sum(axis=0))[0])

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
    The output are the absolute and relative frequencies of the words with the highest relative frequencies in the corpus.
    '''
    def  __init__(self, language):
        super().__init__(language, output_dir='ngram_counts', data_type='csv')
        self.absdf, self.reldf = self.prepare_df() #Absolute and relative word frequencies
        self.modes = [500, 1000, 2000, 5000]

    def prepare_df(self):
        # Get relative word frequcencies
        data_dict = NgramCounter(self.language).load_data(file_name='unigram_full')
        absdf = pd.DataFrame.sparse.from_spmatrix(data=data_dict['dtm'], columns=data_dict['words'], index=data_dict['file_names'])
        # Calculate relative frequencies
        reldf = absdf.divide(absdf.sum(axis=1), axis=0)
        # Sort according to relative frequencies
        reldf = reldf[reldf.sum().sort_values(ascending=False).index]
        return absdf, reldf
    

    def create_data(self, **kwargs):
        mfw = kwargs['mode']
        reldf = self.reldf.iloc[:, :mfw]
        reldf = reldf.sparse.to_dense()
        # Save relative frequencies of words with the highest relative frequencies in the corpus
        self.save_data(data=reldf, file_name=None, file_string='rel', **kwargs)
        self.logger.info(f'Saved {mfw} words with highest relative word frequency to file.')

        # Input for stylo
        absdf = self.absdf[reldf.columns.tolist()]
        absdf = absdf.sparse.to_dense()
        # Save absolute frequencies of words with the highest relative frequencies in the corpus
        self.save_data(data=absdf, file_name=None, file_string='abs', **kwargs)
        

    # def create_filename(self, **kwargs):
    #     file_string = kwargs['file_string']
    #     file_name = super().create_filename(**kwargs, file_string=file_string)
    #     print(file_name)
    #     return file_name
    
# %%
