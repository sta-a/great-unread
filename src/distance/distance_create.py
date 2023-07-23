# %%
# %load_ext autoreload
# %autoreload 2
from sklearn.metrics import pairwise_distances
import pickle
import logging
import time
import os
import math
import pandas as pd
from scipy.spatial.distance import minkowski #################
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/annina/scripts/great_unread_nlp/src")
from feature_extraction.process_d2v import D2vProcessor
from feature_extraction.process_rawtext import NgramCounter
sys.path.insert(1, '/home/annina/scripts/pydelta')
import delta
sys.path.append("..")
from utils import DataHandler, get_doc_paths, get_bookname

logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
ticker_logger = logging.getLogger('matplotlib.ticker')
ticker_logger.setLevel(logging.WARNING)

class Distance(DataHandler):
    def __init__(self, language, output_dir='distance'):
        super().__init__(language, output_dir)

    def create_data(self,**kwargs):
        '''
        Prepare data for distance calculation, call function that calculates distance
        '''
        mode = kwargs['mode']
        distance_metric = self.calculate_distance()
        mx = pairwise_distances(self.df, metric=distance_metric)
        mx = pd.DataFrame(mx, index=self.df.index, columns=self.df.index)
        mx = self.postprocess_mx(mx, mode, dist_to_sim=False)
        self.save_data(data=mx, file_name=None,**kwargs)
        self.logger.info(f'Created similarity matrix.')

    def postprocess_mx(self, mx, mode, dist_to_sim=False):
        if dist_to_sim == True:
            mx = self.distance_to_similarity(mx)
        mx = self.set_diagonal(mx, np.nan)
        self.plot_distance_distribution(mx=mx, mode=mode)
        assert mx.index.equals(mx.columns)
        assert mx.equals(mx.T) # Check if symmetric
        assert not np.any(mx.values == 0) # Test whether any element is 0
        # Check if all diagonal elements of the matrix or DataFrame 'mx' are NaN
        assert np.all(np.isnan(np.diag(mx.values)))        
        non_diag_values = self.get_triangular(mx)
        assert not np.any(np.isnan(non_diag_values)) # Almost the same as next line
        # assert not mx.isnull().values.any()
        # assert mx.all().all()
        return mx

    def set_diagonal(self, mx, value):
        '''
        mx: distance or similarity matrix
        value: the value that should be on the diagonal
        '''
        #df.values[[np.arange(df.shape[0])]*2] = value
        for i in range(0, mx.shape[0]):
            mx.iloc[i, i] = value
        return mx

    def get_triangular(self, mx):
        '''
        mx: symmetric matrix
        Return values in one triangular of the matrix as array. Diagonal is ignored
        '''
        vals = np.tril(mx.values, k=-1).flatten() # Return a copy of an array with elements above the k-th diagonal zeroed
        vals = vals[np.nonzero(vals)] # Remove zeros from tril
        # Check number of elements below the diagonal
        assert len(vals) == self.nr_elements_triangular(mx)
        return vals

    def nr_elements_triangular(self, n_or_mx):
        '''
        Calculate the number of elements in one triangular above the diagonal of a symmetric matrix.
        The diagonal is not counted.
        n(n-1)/2
        '''
        if isinstance(n_or_mx, pd.DataFrame):
            n = n_or_mx.shape[0]
        else:
            n = n_or_mx
        return n*(n-1)/2

    def distance_to_similarity(self, mx):
        '''
        mx: distance matrix
        Invert distances to obtain similarities.
        '''
        def set_diagonal(mx, value):
            for i in range(0, mx.shape[0]):
                mx.iloc[i, i] = value
            return mx
        
        # Assert that there are no Nan
        assert not mx.isnull().values.any()
        # Set diagonal to Nan
        mx = set_diagonal(mx, np.nan)
        # Assert that there are no zero distances
        assert mx.all().all()
        mx = mx.rdiv(1) # divide 1 by the values in the matrix
        return mx

    def plot_distance_distribution(self, mx, mode):
        start = time.time()
        vals = self.get_triangular(mx) # Get triangular matrix, ignore diagonal
        assert not np.any(np.isnan(vals)) # Test whether any element is nan

        # Find most common frequency
        _, counts = np.unique(vals, return_counts=True)
        ind = np.argmax(counts)

        print(f'Minimum {mode}: {min(vals)}. Maximum {mode}: {max(vals)}. Most common {mode}: {vals[ind]}.')

        fig = plt.figure(figsize=(20,6), dpi=300)
        ax = fig.add_subplot(111)

        binsize = 0.001
        xtick_step = 1
        if mode.split('-')[0] == 'cosinedelta':
            binsize = 10
            xtick_step = 50
        #ax.hist(vals, bins = np.arange(0,max(vals) + 0.1, binsize), log=True, ec='black', color='black') #kwargs set edge color
        ax.hist(vals, bins='auto', log=True, ec='black', color='black') #kwargs set edge color

        data_min = min(vals) ###################################
        data_max = max(vals)
        tick_step = math.floor(0.05 * data_max)
        ax.set_xticks(np.arange(0, data_max + tick_step, tick_step))
        # ax.set_xticks(range(min(vals), max(vals)+1, 5))

        ax.set_xlabel(f'{mode.capitalize()}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{mode.capitalize()} distribution')
        #plt.xticks(np.arange(0, max(vals) + 0.1, step=xtick_step))
        plt.xticks(rotation=90)
        ax.grid(False)

        # file_name = self.create_filename(mode=mode, )
        self.save_data(data=plt, data_type='png', mode=mode)
        # plt.savefig('test', format="png")
        print(f'{time.time()-start}s to create {mode} distribution plot.')


class D2vDist(Distance):
    '''
    Create similarity matrices based on doc2vec docuement vectors.
    '''
    def __init__(self, language, output_dir='distance'):
        super().__init__(language, output_dir)
        self.modes = ['doc_tags', 'both_tags']
        self.doc_paths = get_doc_paths(os.path.join(self.data_dir, 'raw_docs', self.language))

    def create_data(self,**kwargs):
        '''
        Create similarity matrix based on d2v embeddings for modes 'doc_tags' and 'both_tags'.
        '''
        mode = kwargs['mode']
        dp = D2vProcessor(self.language, output_dir=None, data_dir=self.data_dir)
        dv_dict = dp.load_data(file_name=f'{mode}.npz')

        # Can only be used for doc_paths, not for chunks
        doc_dvs = {} 
        for doc_path in self.doc_paths:
            book_name = get_bookname(doc_path)
            doc_dvs[book_name] = dv_dict[book_name]
        dv_dict = doc_dvs

        mx = self.calculate_distance(dv_dict)
        mx = self.postprocess_mx(mx, mode, dist_to_sim=False)
        self.save_data(mx, mode=mode)
        self.logger.info(f'Created {mode} similarity matrix.')

    def create_filename(self,**kwargs):
        file_string = f"d2vtpc{self.tokens_per_chunk}"
        return self.create_filename_base(**kwargs, file_string=file_string)

    def calculate_distance(self, dictionary):
        """
        Calculates the pairwise cosine similarity between each pair of vectors in a dictionary.

        Args:
            dictionary (dict): Dictionary mapping names to vectors as numpy arrays.

        Returns:
            mx (pd.DataFrame): Pandas DataFrame representing the pairwise cosine similarity matrix.
        """
        # Extract the names and vectors from the dictionary
        names = list(dictionary.keys())
        vectors = list(dictionary.values())
        # Convert the list of vectors into a 2D numpy array
        vectors = np.array(vectors)
        # Calculate the cosine similarity matrix using sklearn's cosine_similarity function
        # If y is None, the output will be the pairwise similarities between all samples in X.
        mx = cosine_similarity(vectors)
        # Create a DataFrame from the similarity matrix
        mx = pd.DataFrame(mx, index=names, columns=names)
        # Scale values of cosine similarity from [-1, 1] to [0, 1]
        mx = mx.applymap(lambda x: 0.5 * (x + 1))
        return mx
    

class PydeltaDist(Distance):
    def __init__(self, language, output_dir='distance'):
        super().__init__(language, output_dir)
        self.distances = ['cosinedelta', 'burrows', 'eder', 'quadratic']
        self.nmfw_values = [500, 1000, 2000, 5000]
        # self.distances = ['burrows']
        # self.nmfw_values = [500]
        self.modes = [f'{item1}-{item2}' for item1 in self.distances for item2 in self.nmfw_values]
        self.logger.info('Similarity matrices created with delta measures have')

    def create_data(self,**kwargs):
        self.logger.info(f"Distance 'edersimple' is not calculated due to implementation error.")
        mode = kwargs['mode']
        start = time.time()
        mx = self.calculate_distance(mode=mode)
        mx = self.postprocess_mx(mx, mode, dist_to_sim=True)
        self.save_data(mx, mode=mode)
        self.logger.info(f'Created {mode} similarity matrix.')
        print(f'{time.time()-start} for {mode}')

    def get_distance_nmfw(self, mode):
        distance, nmfw = mode.split('-')
        return distance, int(nmfw)

    def get_corpus(self, mode):
        distance, nmfw = self.get_distance_nmfw(mode)
        mfwf = MFWFilter(language=self.language)
        mfwf.file_exists_or_create(mode=nmfw)
        # if not mfwf.file_exists(mode=nmfw):
        #     _ = mfwf.load_all_data()
        corpus = delta.Corpus(mfwf.get_file_path(file_name=None, mode=nmfw))
        return corpus

    def calculate_distance(self, mode):
        distance, nmfw = self.get_distance_nmfw(mode)
        corpus = self.get_corpus(mode=mode)
        if distance == 'burrows':
            mx = delta.functions.burrows(corpus)
        elif distance == 'cosinedelta':
            mx = delta.functions.cosine_delta(corpus)
        elif distance == 'eder':
            mx = delta.functions.eder(corpus)
        elif distance == 'edersimple':
            mx = delta.functions.eder_simple(corpus)
        elif distance == 'quadratic':
            mx = delta.functions.quadratic(corpus)
        return mx
    

class MFWFilter(DataHandler):
    def __init__(self, language):
        super().__init__(language, 'ngram_counts')
        self.modes = [500, 1000, 2000, 5000]
        self.total_unigram_counts, self.book_unigram_mapping = self.load_unigram_counts()

    def load_unigram_counts(self):
        # Load the total unigram counts and book unigram mapping from file.
        nc = NgramCounter(self.language, None, None).load_data(file_name='unigram_counts.pkl')
        book_unigram_mapping = nc['book_unigram_mapping']
        total_unigram_counts = nc['total_unigram_counts']
        total_unigram_counts = pd.DataFrame([total_unigram_counts], index=['counts']).T.sort_values(by='counts', ascending=False)
        return total_unigram_counts, book_unigram_mapping


    def create_filename_base(self, **kwargs):
        data_type = self.get_custom_datatype(**kwargs)
        return f"mfw{str(kwargs['mode'])}.{data_type}"


    def create_data(self,**kwargs):
        """
        Prepare the Most Frequent Words (MFW) DataFrame by filtering the unigram counts.
        
        Finally, it constructs the MFW counts DataFrame
        and saves it to a CSV file for later use.
        """
        self.logger.info('Creating MFW table')
        nmfw = kwargs['mode']
        # MFW based on the total unigram counts
        mfw_set = self.get_mfw_set(nmfw)
        # Filter the book unigram mapping to include only the MFW. 
        book_unigram_mapping_filtered = self.filter_book_unigram_mapping(mfw_set)
        mfw_counts = pd.DataFrame.from_dict(book_unigram_mapping_filtered, orient='index').fillna(0).astype('int64')
        self.save_data(data=mfw_counts, file_name=None, mode=nmfw)
        self.logger.info('Created MFW table')


        # mfw_counts.to_csv('corpustest.csv', header=True, index=True)
        # corpus = delta.Corpus('corpus_words.csv')
        # #corpus.save()
        # print('------------corpus from file----------------------------', corpus)
        # corpus_test = corpus.get_mfw_table(nmfw) ##################################
        # corpus_test.to_csv(f'corpus_test_{nmfw}', header=True, index=True)

    def get_mfw_set(self, nmfw):
        # MFW based on the total unigram counts
        mfw_set = set(self.total_unigram_counts.iloc[:nmfw, :].index.tolist())
        return mfw_set

    def filter_book_unigram_mapping(self, mfw_set):
        """
        Filter the book unigram mapping so that it only contains words in the mfw.
        """
        book_unigram_mapping_filtered = {}
        for filename, book_dict in self.book_unigram_mapping.items():
            book_dict_ = {word: count for word, count in book_dict.items() if word in mfw_set}
            book_unigram_mapping_filtered[filename] = book_dict_
        return book_unigram_mapping_filtered


# %%
