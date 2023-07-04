# %%
# %load_ext autoreload
# %autoreload 2
from sklearn.metrics import pairwise_distances
import pickle
import logging
import os
import pandas as pd
from scipy.spatial.distance import minkowski #################
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import sys
sys.path.append("/home/annina/scripts/great_unread_nlp/src")
from feature_extraction.process_d2v import D2vProcessor
sys.path.insert(1, '/home/annina/scripts/pydelta')
import delta
sys.path.append("..")
from utils import DataHandler

logging.basicConfig(level=logging.DEBUG)
# data_dir = '../../data'



class Distance(DataHandler):
    def __init__(self, language, output_dir):
        super().__init__(language, output_dir)

    def create_all_data(self):
        '''
        Prepare data for distance calculation, call function that calculates distance
        '''
        distance_metric = self.calculate_distance()
        mx = pairwise_distances(self.df, metric=distance_metric)
        mx = pd.DataFrame(mx, index=self.df.index, columns=self.df.index)
        self.logger.info(f'Created similarity matrix.')
        self.save_data(mx)


class D2vDist(Distance):
    '''
    Create similarity matrices based on doc2vec docuement vectors.
    '''
    def __init__(self, language, output_dir='distance'):
        super().__init__(language, output_dir)
        self.modes = ['doc_tags', 'both_tags']

    def create_all_data(self):
        '''
        Create similarity matrix based on d2v embeddings for modes 'doc_tags' and 'both_tags'.
        '''
        for mode in self.modes:
            dp = D2vProcessor(self.language, output_dir=None, data_dir=self.data_dir)
            dp.create_all_data()
            dv_dict = dp.load_data(file_name=f'{mode}.npz')
            simmx = self.calculate_distance(dv_dict)
            self.save_data(simmx, mode=mode)
        self.logger.info(f'Created {mode} similarity matrix.')

    def create_filename(self, kwargs):
        return os.path.join(f"similarity-d2vtpc{self.tokens_per_chunk}-{kwargs['mode']}.{self.data_type}")

    def calculate_distance(self, dictionary):
        """
        Calculates the pairwise cosine similarity between each pair of vectors in a dictionary.

        Args:
            dictionary (dict): Dictionary mapping names to vectors as numpy arrays.

        Returns:
            simmx (pd.DataFrame): Pandas DataFrame representing the pairwise cosine similarity matrix.
        """
        # Extract the names and vectors from the dictionary
        names = list(dictionary.keys())
        vectors = list(dictionary.values())
        # Convert the list of vectors into a 2D numpy array
        vectors = np.array(vectors)
        # Calculate the cosine similarity matrix using sklearn's cosine_similarity function
        # If y is None, the output will be the pairwise similarities between all samples in X.
        simmx = cosine_similarity(vectors)
        # Create a DataFrame from the similarity matrix
        simmx = pd.DataFrame(simmx, index=names, columns=names)
        # Scale values of cosine similarity from [-1, 1] to [0, 1]
        simmx = simmx.applymap(lambda x: 0.5 * (x + 1))
        return simmx

class PydeltaDist(Distance):
    def __init__(self, language, output_dir='distances'):
        super().__init__(language, output_dir)
        self.distances = ['burrows', 'cosinedelta', 'eder', 'quadratic']
        self.mfw_values = [500, 1000, 2000, 5000]

    def create_all_data(self):
        self.logger.info(f"Distance 'edersimple' is not calculated due to implementation error.")
        for mfw in self.mfw_values:
            corpus = self.get_corpus()
            for distance in sorted(self.distances):
                mx = self.calculate_distance(distance, corpus)
                simmx = self.distance_to_similarity(mx)
                self.logger.info(f'Created {distance}{mfw} similarity matrix.')
                self.save_data(simmx, distance=distance, mfw=mfw)

    def get_corpus(self):
        #word_stats = WordStatistics(self.data_dir, self.language, mfw).check_mfwdf_exists()
        #corpus = delta.Corpus(file=word_stats.mfwdf_path)
        path = '/home/annina/scripts/great_unread_nlp/data/ngram_counts/eng/ngram_counts.pkl' ##########################
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file '{path}' does not exist.")
        corpus = delta.Corpus(path)
        return corpus

    def calculate_distance(self, distance, corpus):
        mx = None
        if distance == 'burrows':
            mx = delta.functions.burrows(corpus)
        elif distance == 'cosinedelta':
            mx = delta.functions.cosine_delta(corpus)
        elif distance == 'eder':
            mx = delta.functions.eder(corpus)
        elif distance == 'edersimple':
            mx = delta.functions.eder_simple(self.corpus)
        elif distance == 'quadratic':
            mx = delta.functions.quadratic(corpus)
        return mx

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
    
    def create_filename(self, kwargs):
        return os.path.join(f"similarity-{kwargs['distance']}-{kwargs['mfw']}.{self.data_type}")

class WordStatistics:
    def __init__(self, data_dir, language, nmfw):
        self.data_dir = data_dir
        self.language = language
        self.nmfw = nmfw
        self.wordstat_dir = os.path.join(self.data_dir, f'word_statistics_None', self.language)
        self.wordstat_path = os.path.join(self.wordstat_dir, 'word_statistics.pkl')
        self.mfwdf_path = os.path.join(self.wordstat_dir, f'mfw_{self.nmfw}.csv')

    def check_mfwdf_exists(self):
        """
        Check if the MFW DataFrame file exists. If not, call the `prepare_mfwdf` method to create it.
        """
        if not os.path.exists(self.mfwdf_path):
            self.prepare_mfwdf()

    def load_word_statistics(self):
        """
        Load the word statistics from the pickle file.
        """
        with open(self.wordstat_path, 'rb') as f:
            word_statistics = pickle.load(f)
        return word_statistics

    def prepare_mfwdf(self):
        """
        Prepare the Most Frequent Words (MFW) DataFrame by filtering the word statistics based on MFW criteria.
        
        This method loads the word statistics from the pickle file and retrieves the total unigram counts and
        book unigram mapping. It then identifies the Most Frequent Words (MFW) based on the total unigram counts
        and filters the book unigram mapping to include only the MFW. Finally, it constructs the MFW counts DataFrame
        and saves it to a CSV file for later use.
        """
        word_statistics = self.load_word_statistics()
        total_unigram_counts = word_statistics['total_unigram_counts']
        book_unigram_mapping = word_statistics['book_unigram_mapping']

        mfw = self.get_mfw_set(total_unigram_counts)
        book_unigram_mapping = self.filter_book_unigram_mapping(book_unigram_mapping, mfw)
        mfw_counts = self.get_mfw_counts(book_unigram_mapping)

        mfw_counts.to_csv(self.mfwdf_path, header=True, index=True)

    def get_mfw_set(self, total_unigram_counts):
        """
        Get the set of Most Frequent Words (MFW) based on the total unigram counts.
        """
        mfw = set(
            pd.DataFrame([total_unigram_counts], index=['counts'])
            .T
            .sort_values(by='counts', ascending=False)
            .iloc[:self.nmfw, :]
            .index
            .tolist()
        )
        return mfw

    def filter_book_unigram_mapping(self, book_unigram_mapping, mfw):
        """
        Filter the book unigram mapping based on the Most Frequent Words (MFW) set.
        """
        book_unigram_mapping_ = {}
        for filename, book_dict in book_unigram_mapping.items():
            book_dict_ = {word: count for word, count in book_dict.items() if word in mfw}
            book_unigram_mapping_[filename] = book_dict_
        book_unigram_mapping = book_unigram_mapping_
        return book_unigram_mapping

    def get_mfw_counts(self, book_unigram_mapping):
        """
        Get the Most Frequent Words (MFW) counts DataFrame from the filtered book unigram mapping.
        """
        mfw_counts = pd.concat(
            {k: pd.DataFrame.from_dict(v, 'index').T.reset_index(drop=True) for k, v in book_unigram_mapping.items()},
            axis=0
            ).droplevel(1).fillna(0).astype('int64')
        return mfw_counts



# Example usage:
language = 'eng'
nmfw = 199
function = 'burrows'
dist_name = f'{function}{nmfw}'

# pydelta = PydeltaDist(language, data_dir)
# pydelta.create_all_data()

dd = D2vDist(language)
dd.create_all_data()


# %%
