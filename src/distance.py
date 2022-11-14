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


class NgramDistance(Distance):
    def __init__(self, language, data_dir):
        super().__init__(language, data_dir)
        # nested dict {file_name: {unigram: count}
        self.wordstat_dir = os.path.join(self.data_dir, f'word_statistics_{nr_texts}', language)
        self.unigram_counts_path = os.path.join(self.wordstat_dir, 'book_unigram_mapping.pkl')
        if not os.path.exists(self.unigram_counts_path):
            wordstat_path = os.path.join(self.wordstat_dir, 'word_statistics.pkl')
            word_statistics = self.load_word_statistics(wordstat_path)
            self.unigram_counts_dict = word_statistics['book_unigram_mapping']
            # Delete to save memory
            del word_statistics

            with open(self.unigram_counts_path, 'wb') as f:
                pickle.dump(self.unigram_counts_dict, f, -1)
        else:
            with open(self.unigram_counts_path, 'rb') as f:
                self.unigram_counts_dict = pickle.load(f)
        self.unigram_counts = self.get_unigram_counts_df()


    def load_word_statistics(self, wordstat_path):
        try:
            with open(wordstat_path, 'rb') as f:
                word_statistics = pickle.load(f)
            return word_statistics
        except FileNotFoundError:
            print('Word statistics file does not exist.')


    def get_unigram_counts_df(self):

        start = time.time()
        i=0
        wordcounts_path = os.path.join(self.wordstat_dir, 'wordcounts')
        if not os.path.exists(wordcounts_path):
            os.mkdir(wordcounts_path)
        wordlist = []
        with open(os.path.join(wordcounts_path, 'wordlist.csv'), 'w') as f:
            for file_name, countdict in self.unigram_counts_dict.items():
                print(file_name)
                for word, count in countdict.items():
                    if word not in wordlist:
                        wordlist.append(word)
                        f.write(word + '\n')
                i+=1
                print(i, time.time()-start)
        del wordlist
        print('Time to make word list; ', time.time()-start)
        
        # {file_name: {word: count}}
        start = time.time()
        unigram_counts = pd.concat(
            {k: pd.DataFrame.from_dict(v, 'index').T.reset_index(drop=True, inplace=False) for k, v in self.unigram_counts_dict.items()}, 
            axis=0).droplevel(1).fillna(0).astype('int64')
        unigram_counts.to_csv(os.path.join(self.wordstat_dir, 'unigram_counts.csv'), header=True, index=True)
        print('Time to make df from nested dict: ', time.time()-start)
        return unigram_counts

n = NgramDistance('ger', '../data')
# %%

class PydeltaDist(NgramDistance):
    def __init__(self, language, data_dir):
        super().__init__(language, data_dir)
        self.corpus = self.prepare_corpus()


    def prepare_corpus(self):
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
        dtm_path = str(pathlib.Path(str(self.unigram_counts_path).replace('book_unigram_mapping.pkl', 'corpus_words.csv')).resolve())
        if not os.path.exists(dtm_path):
            self.unigram_counts.T.to_csv(
                    dtm_path,
                    encoding="utf-8",
                    na_rep=0,
                    quoting=csv.QUOTE_NONNUMERIC)

        corpus = delta.Corpus(file=dtm_path)
        return corpus


    def get_corpus(self, n_mfw=500):
        corpus_topn = self.corpus.top_n(n_mfw)
        return corpus_topn



# class Delta(NgramDistance):
#     def __init__(self, language, data_dir, n_mfw=500):
#         super().__init__(language, data_dir)
#         self.n_mfw = n_mfw
#         self.rel_unigram_freq_mfw, self.std_rel_unigram_freq = self.prepare_data()
#         self.df = self.rel_unigram_freq_mfw
    
#     def prepare_data(self):
#         # f_i(D)
#         rel_unigram_freq = self.unigram_counts.div(self.unigram_counts.sum(axis=1), axis=0)
#         # mu_i
#         mean_rel_unigram_freq = rel_unigram_freq.mean(axis=0)
#         # sort according to relative frequency
#         rel_unigram_freq = rel_unigram_freq.reindex(mean_rel_unigram_freq.sort_values(ascending=False).index, axis=1)
#         # select n most common words
#         rel_unigram_freq_mfw = rel_unigram_freq.iloc[:, :self.n_mfw]
#         # sigma_i
#         std_rel_unigram_freq = rel_unigram_freq.std(axis=0).iloc[:self.n_mfw]
#         return rel_unigram_freq_mfw, std_rel_unigram_freq

#     def get_distance_matrix(self):
#         m = self.calculate_dist_matrix()
#         m = pd.DataFrame(m, index=self.df.index, columns=self.df.index)
#         return m


# class ClassicDelta(Delta):
#     # Burrows Delta with Manhattan distance
#     def __init__(self, language, data_dir, n_mfw=500):
#         super().__init__(language, data_dir, n_mfw)

#     def calculate_distance(self, row1, row2):
#         # The callable should take two arrays from X as input and return a value indicating the distance between them.
#         #diff = (abs((row1 - row2).div(self.std_rel_unigram_freq, axis=0)).sum())#/(1/self.n_mfw)
#         diff = np.sum(np.absolute(np.divide(minkowski(row1, row2, p=1), self.std_rel_unigram_freq.values)))
#         return diff
# d = ClassicDelta(language, data_dir)
# matrix = d.get_distance_matrix()

# %%
def is_symmetric(df):
    return df.equals(df.T)

def show_distance_distribution(df):
   values = df.to_numpy()
   # lower triangle of array
   values = np.tril(values).flatten()
   plt.hist(values, bins = np.arange(0,1.1, 0.01))
   plt.show()


language = 'ger' #['eng', 'ger']
data_dir = '../data'
print(delta.functions)

pydelta = PydeltaDist(language, data_dir)
burrows_mtrx = delta.functions.burrows(pydelta.get_corpus(n_mfw=500)) #MFW?
print(is_symmetric(burrows_mtrx))

show_distance_distribution(burrows_mtrx)

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
