from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from importlib.resources import path
import pickle
import os
import pandas as pd
import numpy as np
import csv
import logging
from matplotlib import pyplot as plt
import sys
sys.path.insert(1, '/home/annina/scripts/pydelta')
import delta


class Distance():
    def __init__(self, language, data_dir):
        self.language = language
        self.data_dir = data_dir
        self.df = None
        self.mx = None

    def calculate_distance(self):
        raise NotImplementedError

    def calculate_mx(self, file_name=None):
        self.mx = pairwise_distances(self.df, metric=self.calculate_distance)
        self.mx = pd.DataFrame(self.mx, index=self.df.index, columns=self.df.index)

        if file_name != None:
            self.save_mx(file_name)
        return self.mx

    def save_mx(self, file_name):
        self.mx.to_csv(
            os.path.join(self.data_dir, 'distances', self.language, f'{file_name}.csv'),
            header=True, 
            index=True
        )


class ImprtDistance(Distance):
    '''
    Calculate distance based on feature importances.
    Only use the most important features.
    Weight distances with feature importance.
    '''
    def __init__(self, language, data_dir):
        super().__init__(language, data_dir)
        self.importances_path = os.path.join(self.data_dir, 'importances', self.language,)
        self.df = self.get_df()
        self.importances = self.get_importances()

    def get_df(self):
        best_features = pd.read_csv(
            os.path.join(self.importances_path, f'book_features_best.csv'),
            header=0, 
            index_col='file_name')
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(best_features), columns=best_features.columns, index=best_features.index)
        return df

    def get_importances(self):
        importances = pd.read_csv(
            os.path.join(self.importances_path, f'best_features_importances.csv'),
            header=0)
        return importances.importance.to_numpy()

    def calculate_distance(self, row1, row2):
        # Weighted Euclidean distance
        d = (row1-row2)
        w = self.importances * self.importances
        return np.sqrt((w*d*d).sum())


class WordbasedDistance(Distance):
    '''
    Distances that are based on words (unigrams).
    '''
    def __init__(self, language, data_dir, nmfw, nr_texts):
        # nr_texts is the number of texts for which the features were calculated, None if all features
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

            mfw_counts = pd.concat(
                {k: pd.DataFrame.from_dict(v, 'index').T.reset_index(drop=True, inplace=False) for k, v in book_unigram_mapping_.items()}, 
                axis=0).droplevel(1).fillna(0).astype('int64')
            mfw_counts.to_csv(self.mfw_df_path, header=True, index=True)
        else:
            mfw_counts = pd.read_csv(self.mfw_df_path, header=0)
            print('Loaded mfw table from file.')

    def load_word_statistics(self):
        try:
            with open(self.wordstat_path, 'rb') as f:
                word_statistics = pickle.load(f)
            return word_statistics
        except FileNotFoundError:
            print('Word statistics file does not exist.')


class PydeltaDist(WordbasedDistance):
    '''
    Calculate distances with Pydelta.
    These are the Burrows' Delta and related measures.
    '''
    def __init__(self, language, data_dir, nmfw, nr_texts):
        super().__init__(language, data_dir, nmfw, nr_texts)
        self.corpus = self.get_corpus()
        self.mx = self.calculate_mx()


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


    def calculate_mx(self, file_name=None):
        self.mx = delta.functions.burrows(self.corpus)
        if file_name != None:
            self.save_mx(file_name)
        return self.mx

def is_symmetric(df):
    return df.equals(df.T)

def show_distance_distribution(mx, language):
    values = mx.to_numpy()
    # lower triangle of array
    values = np.tril(values).flatten()
    values = values[values !=0]
    print(len(values))
    if not len(values) == (mx.shape[0]*(mx.shape[0]-1)/2):
        print('Incorrect number of values.')
    print(f'Minimum distance: {min(values)}. Maximum distance: {max(values)}. Nr nonzeros: {np.count_nonzero(values)}')

    fig = plt.figure(figsize=(10,6), dpi=300)
    ax = fig.add_subplot(111)
    ax.hist(values, bins = np.arange(0,max(values) + 0.1, 0.001), log=False)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distance frequency {language}')
    plt.xticks(np.arange(0, max(values) + 0.1, step=0.5))
    plt.xticks(rotation=90)

    plt.show()

def get_importances_mx(language, data_dir):
    #----------------------------------------------
    # Get distance based on feature importance
    #----------------------------------------------
    i = ImprtDistance(language, data_dir)
    imprtmx = i.calculate_mx(file_name='imprtdist')
    show_distance_distribution(imprtmx, language)
    return imprtmx


# %%
def get_pydelta_mx(language, data_dir, nmfw, nr_texts):
    #----------------------------------------------
    # Get Pydelta distances
    #----------------------------------------------
    #print(delta.functions)
    pydelta = PydeltaDist(language, data_dir, nmfw=nmfw, nr_texts=nr_texts)
    #x = corpus.sum(axis=1).sort_values(ascending=False)
    burrowsmx = pydelta.calculate_mx(file_name='burrows')
    # show_distance_distribution(burrowsmx, language)
    print(burrowsmx.simple_score())
    return burrowsmx
    