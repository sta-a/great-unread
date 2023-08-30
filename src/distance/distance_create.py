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
from feature_extraction.embeddings import D2vProcessor
from feature_extraction.ngrams import MfwExtractor
sys.path.insert(1, '/home/annina/scripts/pydelta')
import delta
sys.path.append("..")
from utils import DataHandler, get_bookname

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
        mx = self.min_max_normalization(mx)
        if dist_to_sim == True:
            mx = self.distance_to_similarity(mx)
        print(f'Max similarity: {mx.max().max()}. Min similarity: {mx.min().min()}.')
        mx = self.set_diagonal(mx, 1)
        self.plot_distance_distribution(mx=mx, mode=mode)

        assert mx.index.equals(mx.columns)
        assert mx.equals(mx.T) # Check if symmetric
        self.find_zero_elements(mx)
        # assert not mx.isnull().values.any()
        # assert mx.all().all()
        return mx

    def find_zero_elements(self, mx):
        # Find rows and cols with value 0
        zero_indices = np.where(mx == 0)
        # Get the row and column labels for the zero elements
        rows_with_zeros = mx.index[zero_indices[0]]
        cols_with_zeros = mx.columns[zero_indices[1]]
        # Print the results
        print("\nRow and Column names for Elements that are 0:")
        for row, col in zip(rows_with_zeros, cols_with_zeros):
            print(f"Row: {row}, Column: {col}")
        print('--------------------------------')


    def min_max_normalization(self, mx):
        # Normalize values in matrix to the range between 0 and 1
        min_val = mx.min().min()
        max_val = mx.max().max()
        normmx = (mx - min_val) / (max_val - min_val)
        return normmx

    def distance_to_similarity(self, distmx):
        '''
        distmx: normalized distance matrix
        Invert distances to obtain similarities.
        '''        
        # Assert that there are no Nan
        assert not distmx.isnull().values.any()
        # Assert that there are no zero distances (ignore the diagonal)
        values = self.get_triangular(distmx)
        assert not np.any(values == 0)
        simmx = 1 - distmx
        return simmx

    def set_diagonal(self, mx, value):
        '''
        mx: distance or similarity matrix
        value: the value to set the diagonal to
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
        # Get the indices of the upper triangle of the matrix
        indices = np.triu_indices(n=mx.shape[0], k=1)
        # Access the elements in the upper triangle using the obtained indices
        vals = mx.values[indices].tolist()
        # Check number of elements below the diagonal
        assert int(len(vals)) == self.nr_elements_triangular(mx)
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
        return int(n*(n-1)/2)

    def plot_distance_distribution(self, mx, mode):
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
        # ax.set_xticks(np.arange(0, data_max + tick_step, tick_step))
        # ax.set_xticks(range(min(vals), max(vals)+1, 5))

        ax.set_xlabel(f'Similarity')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{mode.capitalize()} distribution')
        #plt.xticks(np.arange(0, max(vals) + 0.1, step=xtick_step))
        plt.xticks(rotation=90)
        ax.grid(False)

        # file_name = self.create_filename(mode=mode, )
        self.save_data(data=plt, data_type='svg', mode=mode)
        plt.close()
        # plt.savefig('test', format="svg")


class D2vDist(Distance):
    '''
    Create similarity matrices based on doc2vec docuement vectors.
    '''
    def __init__(self, language, output_dir='distance'):
        super().__init__(language, output_dir)
        self.modes = ['doc_tags', 'both_tags']


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
        mx.to_csv(os.path.join('/home/annina/scripts/great_unread_nlp/data/distance/', self.language, 'distmx', f'{mode}.csv')) ############################
        mx = self.postprocess_mx(mx, mode, dist_to_sim=False)
        self.save_data(mx, mode=mode)
        self.logger.info(f'Created {mode} similarity matrix.')

    def create_filename(self,**kwargs):
        file_string = f"d2vtpc{self.tokens_per_chunk}"
        file_name = super().create_filename(**kwargs, file_string=file_string)
        return file_name

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
        self.distances = ['burrows', 'cosinedelta', 'eder', 'quadratic']
        self.nmfw_values = [500, 1000, 2000, 5000]
        # self.distances = ['burrows']
        # self.nmfw_values = [500]
        self.modes = [f'{item1}-{item2}' for item1 in self.distances for item2 in self.nmfw_values]

    def create_data(self,**kwargs):
        self.logger.info(f"Distance 'edersimple' is not calculated due to implementation error.")
        mode = kwargs['mode']
        mx = self.calculate_distance(mode=mode)
        mx.to_csv(os.path.join('/home/annina/scripts/great_unread_nlp/data/distance/', self.language, 'distmx', f'{mode}.csv'))
        mx = self.postprocess_mx(mx, mode, dist_to_sim=True)
        self.save_data(mx, mode=mode)
        self.logger.info(f'Created {mode} similarity matrix.')

    def get_params_from_mode(self, mode):
        distance, nmfw = mode.split('-')
        return distance, int(nmfw)

    def get_corpus(self, mode):
        distance, nmfw = self.get_params_from_mode(mode)
        mfwf = MfwExtractor(language=self.language)
        mfwf.file_exists_or_create(mode=nmfw)
        corpus = delta.Corpus(mfwf.get_file_path(file_name=None, mode=nmfw))
        return corpus

    def calculate_distance(self, mode):
        distance, nmfw = self.get_params_from_mode(mode)
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


class StyloDistance(PydeltaDist):
    '''
    Class for turning Stylo distances into similarities and comparing them with other delta implementations.
    '''
    def __init__(self, language, output_dir='distance'):
        super().__init__(language, output_dir) 
        self.output_dir = os.path.join(self.output_dir, 'stylo')
        self.distances = ['burrows', 'eder', 'edersimple', 'linear']
        self.nmfw_values = [500, 1000, 2000, 5000]
        # self.distances = ['burrows']
        # self.nmfw_values = [500]
        self.modes = [f'{item1}-{item2}' for item1 in self.distances for item2 in self.nmfw_values]

    def create_input_filename(self,**kwargs):
        file_name = super().create_filename(**kwargs, file_string='dist')
        return file_name
       

    def create_data(self, **kwargs):
        input_filename = self.create_input_filename(**kwargs)
        print(input_filename)
        mx = self.load_data(file_name=input_filename)
        mode = kwargs['mode']
        mx = self.postprocess_mx(mx, mode, dist_to_sim=True)
        self.save_data(mx, mode=mode)
        self.logger.info(f'Created {mode} similarity matrix.')
    

d = StyloDistance(language='eng')
d.load_all_data()

# %%
