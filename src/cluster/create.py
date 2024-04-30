# %%
import pickle
import logging
import time
import re
import os
import math
from copy import deepcopy
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import minkowski, squareform, cosine
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from feature_extraction.embeddings import D2vProcessor
from feature_extraction.ngrams import MfwExtractor
from utils import DataHandler, get_filename_from_path

logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
ticker_logger = logging.getLogger('matplotlib.ticker')
ticker_logger.setLevel(logging.WARNING)


class SimMx(DataHandler):
    '''
    Class for distance/smiliarity matrices
    '''
    def __init__(self, language, name=None, mx=None, normalized=True, is_sim=True, is_directed=False, is_condensed=False, has_dmx=True, output_dir='similarity'):

        # Check input
        if not is_sim and not has_dmx:
            raise ValueError('If is_sim is False, has_dmx must be True.')

        super().__init__(language, output_dir)
        self.add_subdir('simmxs')
        self.name = name
        self.mx = mx
        self.normalized = normalized
        self.min_max_normalization()
        self.is_sim = is_sim
        self.is_directed = is_directed
        self.is_condensed = is_condensed
        self.has_dmx = has_dmx

        if self.is_sim:
            if self.has_dmx:
                self.dmx = self.invert_mx()
        else:
            self.dmx = self.mx
            assert self.has_dmx
            self.mx = self.invert_mx()
            self.is_sim = True
        
        assert self.is_sim


    def invert_mx(self, to_dist=True):
        '''
        Invert similarities to obtain distances and vice versa.
        '''
        assert self.normalized, f'Matrix must be normalized before it is turned into similarity.'
        if to_dist:
            mx = deepcopy(self.mx)
        else:
            mx = deepcopy(self.dmx)

        # non_null_values = mx[mx.isnull().values]
        # Assert that there are no Nan
        assert not mx.isnull().values.any()
        # Assert that there are no zero similarities (ignore the diagonal)
        values = self.get_triangular()
        assert not np.any(values == 0)
        mx = 1 - mx
        # assert np.all(np.diagonal(mx) == diag_val), 'Not all values on the diagonal are 1.'
        return mx


    def postprocess_mx(self, **kwargs):
        self.set_diagonal(value=1)
        self.plot_similarity_distribution(**kwargs)

        # Sort rows and columns alphabetically
        self.mx = self.mx.sort_index(axis=0)
        self.mx = self.mx.sort_index(axis=1)

        assert self.mx.index.equals(self.mx.columns)
        assert self.mx.equals(self.mx.T) # Check symmetry
        # Check if 1 is only on the diagonal
        # assert not np.any((self.mx.values != np.eye(len(self.mx))) & (self.mx.values == 1))
        print('self.mx.isnull().values.any()', self.mx.isnull().values.any())
        print('self.mx.all().all()', self.mx.all().all()) # Check if all values are True


    def find_zero_elements(self):
        # Find rows and cols with value 0
        zero_indices = np.where(self.mx == 0)
        # Get the row and column labels for the zero elements
        rows_with_zeros = self.mx.index[zero_indices[0]]
        cols_with_zeros = self.mx.columns[zero_indices[1]]
        # Print the results
        if len(rows_with_zeros) > 0 or len(cols_with_zeros) > 0:
            print("\nRow and Column names for Elements that are 0:")
            for row, col in zip(rows_with_zeros, cols_with_zeros):
                print(f"Row: {row}, Column: {col}")
            print('--------------------------------')


    def min_max_normalization(self):
        mx = self.mx
        if not self.normalized:
            # Normalize values in matrix to the range between 0 and 1
            min_val = mx.min().min()
            max_val = mx.max().max()
            mx = (mx - min_val) / (max_val - min_val)
            print(f'After normalization: Max similarity: {mx.max().max()}. Min similarity: {mx.min().min()}.')
            self.normalized = True
        self.mx = mx


    def dist_to_condensed(self):
        '''
        Condensed matrix, input for scipy.cluster.hierarchy.linkage.
        '''
        assert self.has_dmx
        dmx = deepcopy(self.dmx)
        if not self.is_condensed:
            dmx = squareform(dmx, checks=True)
            self.is_condensed = True
        self.dmx = dmx


    def set_diagonal(self, value):
        '''
        mx: distance or similarity matrix
        value: the value to set the diagonal to
        '''
        #df.values[[np.arange(df.shape[0])]*2] = value
        mx = self.mx
        for i in range(0, mx.shape[0]):
            mx.iloc[i, i] = value
        self.mx = mx

    def get_triangular(self):
        '''
        mx: symmetric matrix
        Return values in one triangular of the matrix as array. Diagonal is ignored
        '''
        # Get the indices of the upper triangle of the matrix
        indices = np.triu_indices(n=self.mx.shape[0], k=1)
        # Access the elements in the upper triangle using the obtained indices
        vals = self.mx.values[indices].tolist()
        # Check number of elements below the diagonal
        assert int(len(vals)) == self.nr_elements_triangular(self.mx)
        return vals

    def nr_elements_triangular(self, n_or_mx):
        '''
        n_or_mx: pass either a number (n) or a matrix with shape (n,n)
        Calculate the number of elements in one triangular above the diagonal of a symmetric matrix.
        The diagonal is not counted.
        nr_elements = n(n-1)/2
        '''
        if isinstance(n_or_mx, pd.DataFrame):
            n = n_or_mx.shape[0]
        else:
            n = n_or_mx
        return int(n*(n-1)/2)


    def plot_similarity_distribution(self, **kwargs):
        vals = self.get_triangular() # Get triangular matrix, ignore diagonal
        assert not np.any(np.isnan(vals)) # Test whether any element is nan

        # Find most common frequency
        _, counts = np.unique(vals, return_counts=True)
        ind = np.argmax(counts)

        print(f'Minimum {self.name}: {min(vals)}. Maximum {self.name}: {max(vals)}. Most common {self.name}: {vals[ind]}.')

        fig = plt.figure(figsize=(20,6), dpi=300)
        ax = fig.add_subplot(111)

        binsize = 0.001
        xtick_step = 1
        if self.name.split('-')[0] == 'cosinedelta':
            binsize = 10
            xtick_step = 50
        #ax.hist(vals, bins = np.arange(0,max(vals) + 0.1, binsize), log=True, ec='black', color='black')
        ax.hist(vals, bins='auto', log=True, ec='black', color='black')

        if self.is_sim:
            xlabel = 'Similarity'
        else:
            xlabel = 'Distance'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{self.name.capitalize()} distribution')
        #plt.xticks(np.arange(0, max(vals) + 0.1, step=xtick_step))
        plt.xticks(rotation=90)
        ax.grid(False)


        # Pass file name as mode instead of file name because kwargs might contain more information about the file name
        # This way, the kwargs are dealt with automatically
        self.save_data(data=plt, subdir=True, data_type='svg', file_name=f'{self.name}', **kwargs)
        plt.close()


class SimMxCreatorBase(DataHandler):
    '''
    Base class for calculating distance matrices
    '''
    def __init__(self, language, output_dir='similarity'):
        super().__init__(language, output_dir)
        self.add_subdir('simmxs')

    def calculate_similarity(self):
        pass

    def create_data(self,**kwargs):
        '''
        Prepare data for similarity calculation, call function that calculates similarity
        '''
        mode = kwargs.get('mode')
        mx = pairwise_distances(self.df, metric=self.calculate_similarity)
        mx = pd.DataFrame(mx, index=self.df.index, columns=self.df.index)
        sm = SimMx(language=self.language, name=mode, mx=mx, normalized=False, is_sim=False)
        sm.postprocess_mx()
        self.save_data(data=sm.mx, file_name=sm.name, subir=True, use_kwargs_for_fn='mode', pandas_kwargs={'index': True}, **kwargs)
        self.logger.debug(f'Created similarity matrix.')


    def load_data(self, load=True, file_name=None, **kwargs):
        '''
        Load mx from csv and create SimMx object
        '''
        mx = super().load_data(load, file_name, **kwargs)
        file_name = get_filename_from_path(self.get_file_path(file_name, **kwargs))
        mx = SimMx(self.language, name=file_name, mx=mx, normalized=True, is_sim=True)
        return mx

    

class D2vDist(SimMxCreatorBase):
    '''
    Create similarity matrices based on doc2vec docuement vectors.
    '''
    def __init__(self, language, output_dir='similarity', modes=['full', 'both']):
        super().__init__(language, output_dir=output_dir)
        self.modes = modes
        self.file_string = 'd2v'


    def get_embeddings_dict(self, mode):
        dp = D2vProcessor(language=self.language, tokens_per_chunk=self.tokens_per_chunk)

        # Can only be used for doc_paths, not for chunks
        doc_dvs = {} 
        for doc_path in self.doc_paths:
            book_name = get_filename_from_path(doc_path)
            d2v_embeddings = dp.load_data(file_name=book_name, mode=mode, subdir=True)[book_name]
            doc_dvs[book_name] = d2v_embeddings
        emb_dict = doc_dvs
        return emb_dict
    

    def create_data(self, **kwargs):
        '''
        Create similarity matrix based on d2v embeddings for modes 'doc_tags' and 'both'.
        '''
        mode = kwargs.get('mode')
        emb_dict = self.get_embeddings_dict(mode)
        mx = self.calculate_similarity(emb_dict)
        sm = SimMx(language=self.language, name=mode, mx=mx, normalized=False, is_sim=True)
        sm.postprocess_mx(file_string=self.file_string)
        # Scale values of cosine similarity from [-1, 1] to [0, 1]
        # mx = mx.applymap(lambda x: 0.5 * (x + 1))
        self.save_data(data=sm.mx, subdir=True, mode=sm.name, file_string=self.file_string, use_kwargs_for_fn='mode', pandas_kwargs={'index': True})
        self.logger.debug(f'Created {mode} similarity matrix.')


    def calculate_similarity(self, dictionary):
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
        return mx
    

class DistanceMetrics():
    '''
    Delta Distance metrics.
    Some are duplicated to check different implementations.
    If possible, output is compared to Stylo output.
    '''
    def __init__(self):
        self.registry = {}
        self.register_functions()
    
    def register(self, name, func, alias=None, duplicated=False):
        self.registry[name] = {'func': func, 'alias': alias, 'duplicated': duplicated}
    
    def get_func(self, name):
        return self.registry[name]['func']
    
    def get_alias(self, name):
        return self.registry[name]['alias']
    
    def is_duplicated(self, name):
        return self.registry[name]['duplicated']
    
    def register_functions(self):
        # Sources: Stylo, Evert2017, Jannidis2015 (Pydelta)
        # Register stylo function names as aliases
        self.register('burrows', self.burrows, 'burrows')
        self.register('burrows_argamon', self.burrows_argamon, 'burrows', True)
        self.register('eder', self.eder, 'eder')
        self.register('edersimple', self.edersimple, 'edersimple')
        self.register('argamonquadratic', self.argamonquadratic, 'argamon')
        self.register('argamonquadratic_argamon', self.argamonquadratic_argamon, 'argamon', True)
        self.register('argamonlinear', self.argamonlinear) # Not implemented in stylo
        self.register('cosinedelta', self.cosinedelta, 'cosinedelta')
        self.register('cosinesim', self.cosinesim) # Implemented in Stylo as 'dist.cosine', but code has bug
        self.register('minmax', self.minmax) # Implemented in Stylo as 'minmax', but too slow for big matrix. Tested with toy example, see minmax dir
        self.register('canberra', 'canberra')
        self.register('braycurtis', 'braycurtis')
        self.register('chebyshev', 'chebyshev')
        self.register('sqeuclidean', 'sqeuclidean')
        self.register('correlation', 'correlation')
        self.register('manhattan', 'manhattan') # Manhattan distance without z-score transformation

    def burrows(self, row1, row2, std_dev):
            assert len(row1) == len(row2) == len(std_dev)
            dist = minkowski(row1, row2, p=1, w=(1/std_dev))
            return dist/(len(row1))
        
    def burrows_argamon(self, row1, row2, std_dev):
    # Alternative implementation of self.burrows()
    # Delta calculated according to simplified formula in Argamon2008
    # Argamon discards scaling with 1/n, here it is done to maintain classic Delta formula
        diff = abs(row1-row2)
        diff = diff / std_dev
        return diff.mean()
    
    def eder(self, row1, row2):
        # Eder's Delta (Eder2015) makes use of a ranking factor that reduces the weight of less-frequent words' z-scores
        # row1 and row2 contain z-scores
        n = len(row1)
        rf = np.arange(n, 0, -1) / n
        row1 = row1 * rf
        row2 = row2 * rf
        dist = np.sum(np.abs(row1 - row2))
        return dist
    
    def edersimple(self, row1, row2):
        row1 = np.sqrt(row1)
        row2 = np.sqrt(row2)
        dist = np.sum(np.abs(row1 - row2))
        return dist

    def argamonquadratic(self, row1, row2):
        # Argamon’s Quadratic Delta
        # Follow notation in Stylo and Jannidis2015
        # Stylo implelentation: euclidean distance between z-scores
        # row1 and row2 are already transformed to z-scores
        assert len(row1) == len(row2)
        dist = minkowski(row1, row2, p=2)
        return dist/len(row1)

    def argamonquadratic_argamon(self, row1, row2, std_dev):
        # Alternative implementation of self.argamon()
        # Argamon’s Quadratic Delta
        # Argamon's notation with simplified formula that does not need z-scores: use word frequencies normalized by inversed squared std dev
        # Follow notation in Argamon2008
        dist = minkowski(row1, row2, p=2, w=(1/np.square(std_dev)))
        return dist/len(row1)
    
    def argamonlinear(self, row1, row2, b):
        # Not implemented in Stylo
        # Partially implemented in Pydelta
        return minkowski(row1, row2, p=1, w=1/b)

    def cosinesim(self, df):
        # Cosine similarity between pairs of rows
        return cosine_similarity(df)

    def minmax(self, row1, row2):
        # Ruzicka Distance, Weighted Jaccard distance
        row1 = row1.reshape(-1, 1)
        row2 = row2.reshape(-1, 1)
        q = np.concatenate([row1,row2], axis=1)
        sim = np.sum(np.amin(q,axis=1))/np.sum(np.amax(q,axis=1))
        return 1 - sim

    def cosinedelta(self, row1, row2):
        # Angular delta (Evert2017, Smith & Aldridge 2011)
        return cosine(row1, row2)



class Delta(SimMxCreatorBase):
    def __init__(self, language):
        super().__init__(language)
        # self.add_subdir('delta_distances')
        self.input_dir = os.path.join(self.data_dir, 'ngram_counts', self.language)
        self.input_dfs = self.load_mfw_dfs()
        self.nmfws = list(reversed(self.input_dfs.keys())) # Reverse so that small numbers come first for faster calculation during testing
        self.metrics = DistanceMetrics()
        self.modes = [f'{metric}-{nmfw}' for metric in self.metrics.registry.keys() for nmfw in self.nmfws]
        self.n_jobs = 1


    def load_mfw_dfs(self):
        # Load all files with format rel-xxx.csv, where xxx represents the MFW
        # Load relative frequencies of MFW
        pattern = re.compile(r'rel-(\d+)\.csv')
        dfs = {}

        # Iterate over files in the directory
        for file_name in os.listdir(self.input_dir):
            match = pattern.match(file_name)
            if match:
                file_path = os.path.join(self.input_dir, file_name)
                df = pd.read_csv(file_path)
                df = df.set_index('file_name', inplace=False)
                # Extract nmfw from file name
                nmfw = re.split(r'[-.]', file_name)[1]
                dfs[nmfw] = df
        return dfs

              
    def calculate_similarity(self, metric, df):
        # Z-Score
        if metric in ['eder', 'argamonquadratic', 'cosinedelta']:
            df = zscore(df, axis=0)
            mx = pd.DataFrame(pairwise_distances(df.values, metric=self.metrics.get_func(metric), n_jobs=self.n_jobs), index=df.index, columns=df.index)
        
        # Eder's weighting factor
        elif metric in ['argamonlinear']:
            # Calculate b for linear Delta
            x = df - df.median()
            b = x.abs().sum() / len(df.columns) # Sum over all documents and divide by nmfw
            mx = pd.DataFrame(pairwise_distances(df.values, metric=self.metrics.get_func(metric), n_jobs=self.n_jobs, b=b), index=df.index, columns=df.index)
        
        # Std dev
        elif metric in ['burrows', 'burrows_argamon', 'argamonquadratic_argamon']:
            std_dev = df.std(axis=0)
            mx = pd.DataFrame(pairwise_distances(df.values, metric=self.metrics.get_func(metric),  n_jobs=self.n_jobs, std_dev=std_dev), index=df.index, columns=df.index)
        
        # No additional input, or metric function from sklearn
        elif metric in ['edersimple', 'minmax', 'canberra', 'braycurtis', 'chebyshev', 'sqeuclidean', 'correlation', 'manhattan']:
            mx = pd.DataFrame(pairwise_distances(df.values, metric=self.metrics.get_func(metric), n_jobs=self.n_jobs), index=df.index, columns=df.index)
        
        # Calculation on whole df instead of rows
        elif metric in ['cosinesim']:
            func = self.metrics.get_func(metric)
            mx = pd.DataFrame(data=func(df), index=df.index, columns=df.index)
        
        else:
            raise NotImplementedError(f"Metric '{metric}' is not supported.")
        return mx


    def create_data(self, **kwargs):
        mode = kwargs.get('mode')

        metric, nmfw = mode.split('-')
        print(mode, metric, nmfw)
        df = self.input_dfs[nmfw]
        mx = self.calculate_similarity(metric, df)
        self.compare_with_stylo(mode, mx)

        sm = SimMx(language=self.language, name=mode, mx=mx, normalized=False, is_sim=False)
        sm.postprocess_mx()
        self.save_data(data=sm.mx, subdir=True, mode=sm.name, use_kwargs_for_fn='mode', pandas_kwargs={'index': True})
        self.logger.debug(f'Created {mode} similarity matrix.')



    def compare_with_stylo(self, mode, mx):
        print(f'\n\n---------------------------\nChecking mode {mode}')
        metric, nmfw = mode.split('-')
        alias = self.metrics.get_alias(metric)

        alias = f'{alias}-{nmfw}'
        print(f'Comparing {metric} to stylo file {alias}')
        alias = alias + '.csv'
        stylo_dir = os.path.join(self.output_dir, 'stylo_distances')
        stylo_files = set(os.listdir(stylo_dir))

        if alias in stylo_files:
            stylo_file = os.path.join(stylo_dir, alias)

            # Compare file contents
            stylo_df = pd.read_csv(stylo_file, header=0, index_col=0)

            # Check if headers and indices are the same
            assert all(mx.columns == stylo_df.columns) and all(mx.index == stylo_df.index)

            # Compare values (allowing for rounding errors)
            diff = abs(mx.values - stylo_df.values)
            
            # Calculate 2 percent of the maximum value
            tolerance_percent = 0.05
            tolerance = tolerance_percent*mx.max().max()

            assert diff.max() <= tolerance, f'{metric}, {alias}: Stylo and Delta values are different.'
            print(f'{metric}, {alias}: Stylo and {self.__class__.__name__} produce the same results within a tolerance of {100*tolerance_percent} %.')
        else:
            print(f'{metric}, {alias} not found in Stylo files.')


    def load_all_data(self, **kwargs):
        # Check if file exists, create it if necessary, return all data
        all_data = {}
        # If load_all is False, only load data for mxs that are not duplicated
        modes = [x for x in self.modes if self.metrics.is_duplicated(f'{x.split("-")[0]}') is False]
        for mode in modes:
            data  = self.load_data(load=True, mode=mode, **kwargs)
            all_data[mode] = data
        return all_data


# %%
