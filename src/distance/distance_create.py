from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from importlib.resources import path
import pickle
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(1, '/home/annina/scripts/pydelta')
import delta
from scipy.spatial.distance import minkowski



class Distance():
    def __init__(self, language, data_dir):
        self.language = language
        self.data_dir = data_dir
        self.df = None ####################
        self.mx = None ###################################

    def calculate_distance(self):
        raise NotImplementedError

    def calculate_mx(self, file_name=None):
        self.mx = pairwise_distances(self.df, metric=self.calculate_distance)
        self.mx = pd.DataFrame(self.mx, index=self.df.index, columns=self.df.index)

        if file_name != None:
            self.save_mx(self.mx, file_name)
        return self.mx

    def save_mx(self, mx, file_name):
        mx = mx.sort_index(axis=0).sort_index(axis=1)
        mx.to_csv(
            os.path.join(self.data_dir, 'distances', self.language, f'distances_{file_name}.csv'),
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
    def __init__(self, language, data_dir, nmfw):
        super().__init__(language, data_dir)
        self.nmfw = nmfw
        self.wordstat_dir = os.path.join(self.data_dir, f'word_statistics_None', language)
        self.wordstat_path = os.path.join(self.wordstat_dir, 'word_statistics.pkl')
        self.mfw_df_path = os.path.join(self.wordstat_dir, f'mfw_{self.nmfw}.csv')
        self.prepare_mfw_df()


    def prepare_mfw_df(self):

        if not os.path.exists(self.mfw_df_path):
            word_statistics = self.load_word_statistics()
            print(f'Preparing {self.nmfw} mfw table')

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
        # pydelta.Corpus accepts string
        corpus = delta.Corpus(file=self.mfw_df_path)
        return corpus

    def calculate_mx(self, function, file_name=None):
        mx = None
        if function == 'burrows':
            mx = delta.functions.burrows(self.corpus)
        elif function == 'quadratic':
            mx = delta.functions.quadratic(self.corpus)
        elif function == 'eder':
            mx = delta.functions.eder(self.corpus)
        elif function == 'edersimple':
            mx = delta.functions.eder_simple(self.corpus)
        elif function == 'cosinedelta':
            mx = delta.functions.cosine_delta(self.corpus)

        if file_name != None:
            self.save_mx(mx, file_name)
        return mx


def get_importances_mx(language, data_dir):
    i = ImprtDistance(language, data_dir)
    mx = i.calculate_mx(file_name='imprtdist')
    #plot_distance_distribution(mx, language, 'imprt', data_dir)
    return mx


# %%
def create_pydelta_mx(language, data_dir, **kwargs):
    #print(delta.functions)
    dist_name = kwargs['dist_name']
    nmfw = kwargs['nmfw']
    function = kwargs['function']
    pydelta = PydeltaDist(language, data_dir, nmfw=nmfw)
    #x = corpus.sum(axis=1).sort_values(ascending=False)
    mx = pydelta.calculate_mx(function, file_name=dist_name)
    #plot_distance_distribution(mx, language, dist_name, data_dir)
    # print(mx.simple_score())
    return mx
    

def load_distance_mx(language, data_dir, **kwargs):
    dist_name = kwargs['dist_name']
    mx_path = os.path.join(data_dir, 'distances', language, f'distances_{dist_name}.csv')
    if os.path.exists(mx_path):
        print(f'Loading {dist_name} distance matrix from file.')
        mx = pd.read_csv(mx_path, header=0, index_col=0)
        print(f'Loaded {dist_name} distance matrix from file.')
    else:
        mx = None
        print(f'Creating {dist_name} distance matrix.')
        if dist_name == 'imprt':
            mx = get_importances_mx(language, data_dir)
        else:
            mx = create_pydelta_mx(language, data_dir, **kwargs)
        print(f'Created {dist_name} distance matrix.')
        # Sort columns and index alphabetically
        mx = mx.sort_index(axis=0).sort_index(axis=1)

    return mx
