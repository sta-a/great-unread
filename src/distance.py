# %%
from importlib.resources import path
import pickle
import os
import pathlib
import pandas as pd
import numpy as np
import csv
import logging
from scipy.spatial.distance import minkowski
from sklearn.metrics import pairwise_distances
import sys
sys.path.insert(1, '/home/annina/scripts/pydelta')
import delta
from sklearn_extra.cluster import KMedoids
from matplotlib import pyplot as plt
import time

nr_texts = None
language = 'ger' #['eng', 'ger']
data_dir = '../data'
nmfw = 500
print(delta.functions)

class Distance():
    def __init__(self, language, data_dir):
        self.language = language
        self.data_dir = data_dir
        self.df = None

    def calculate_distance(self):
        raise NotImplementedError

    def calculate_dist_matrix(self):
        self.dist_matrix = pairwise_distances(self.df, metric=self.calculate_distance)
        return self.dist_matrix


class WordbasedDistance(Distance):
    def __init__(self, language, data_dir, nmfw):
        super().__init__(language, data_dir)
        self.nmfw = nmfw
        self.wordstat_dir = os.path.join(self.data_dir, f'word_statistics_{nr_texts}', language)
        self.wordstat_path = os.path.join(self.wordstat_dir, 'word_statistics.pkl')
        self.mfw_df_path = os.path.join(self.wordstat_dir, f'mfw_{self.nmfw}.csv')
        self.prepare_mfw_df()


    def prepare_mfw_df(self):

        if not os.path.exists(self.mfw_df_path):
            word_statistics = self.load_word_statistics()

            total_unigram_counts = word_statistics['total_unigram_counts']
            # nested dict {file_name: {unigram: count}
            book_unigram_mapping = word_statistics['book_unigram_mapping']
            # Delete to save memory

            mfw = set(
                    pd.DataFrame([total_unigram_counts], index=['counts']) \
                    .T \
                    .sort_values(by='counts', ascending=False) \
                    .iloc[:self.nmfw, :] \
                    .index \
                    .tolist()
                )

            # keep only counts of the mfw for each book
            book_unigram_mapping_ = {}
            # {file_name: {word: count}}
            for filename, book_dict in book_unigram_mapping.items():
                book_dict_ = {}
                for word in mfw:
                    if word in book_dict:
                        book_dict_[word] = book_dict[word]
                book_unigram_mapping_[filename] = book_dict_
            del word_statistics
            del total_unigram_counts
            del book_unigram_mapping


            start = time.time()
            mfw_counts = pd.concat(
                {k: pd.DataFrame.from_dict(v, 'index').T.reset_index(drop=True, inplace=False) for k, v in book_unigram_mapping_.items()}, 
                axis=0).droplevel(1).fillna(0).astype('int64')
            mfw_counts.to_csv(self.mfw_df_path, header=True, index=True)
            print('Time to make df from nested dict: ', time.time()-start)
        else:
            mfw_counts = pd.read_csv(self.mfw_df_path, header=0)
            print('Loaded mfw table from file.')
        print(mfw_counts)

    def load_word_statistics(self):
        try:
            with open(self.wordstat_path, 'rb') as f:
                word_statistics = pickle.load(f)
            return word_statistics
        except FileNotFoundError:
            print('Word statistics file does not exist.')


class PydeltaDist(WordbasedDistance):
    def __init__(self, language, data_dir, nmfw):
        super().__init__(language, data_dir, nmfw)
        self.corpus = self.get_corpus()


    def get_corpus(self):
        """
        Saves the corpus to a CSV file.

        The corpus will be saved to a CSV file containing documents in the
        columns and features in the rows, i.e. a transposed representation.
        Document and feature labels will be saved to the first row or column,
        respectively.

        Args:
            filename (str): The target file.
        """
        # pydelta.Corpus takes string
        corpus = delta.Corpus(file=self.mfw_df_path)
        return corpus


# %%
def is_symmetric(df):
    return df.equals(df.T)

def show_distance_distribution(df, nmfw):
    values = df.to_numpy()
    # lower triangle of array
    values = np.tril(values).flatten()
    values = values[values !=0]
    print(len(values))
    if not len(values) == (nmfw*(nmfw-1)/2):
        print('Incorrect number of values.')
    print(f'Minimum distance: {min(values)}. Maximum distance: {max(values)}.')
    plt.hist(values, bins = np.arange(0,max(values) + 0.1, 0.001))
    plt.show()


pydelta = PydeltaDist(language, data_dir, nmfw=nmfw)
pydelta_corpus = pydelta.get_corpus()

#x = corpus.sum(axis=1).sort_values(ascending=False)
burrows_mtrx = delta.functions.burrows(pydelta_corpus) #MFW?
print(is_symmetric(burrows_mtrx))

show_distance_distribution(burrows_mtrx, nmfw)

# %%
# kmedoids = KMedoids(n_clusters=2, metric='precomputed', method='pam', init='build', random_state=8).fit(burrows_mtrx)
from sklearn.manifold import MDS
X_transform = MDS(n_components=2, dissimilarity='precomputed', random_state=8).fit_transform(burrows_mtrx)

colors = ['r', 'g', 'b', 'c', 'm']
size = [64, 64, 64, 64, 64]
fig = plt.figure(2, (10,4))
ax = fig.add_subplot(122)
plt.scatter(X_transform[:,0], X_transform[:,1], s=size, c=colors)
plt.title('Embedding in 2D')
fig.subplots_adjust(wspace=.4, hspace=0.5)
plt.show()

# %%

# %%

# distance, dissimilarity'
# Std of whole corpus or only mfw????
# function registry
#similarity or dissimiliarity
# use all distances

# agglomerative hierarchical clustering, k-means, or density-based clustering (DBSCAN)
