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
import scipy.cluster.hierarchy as sch
sys.path.insert(1, '/home/annina/scripts/pydelta')
import delta
from hpo_functions import get_author_groups, get_data
import matplotlib as mpl
from sklearn.manifold import MDS

class ClusterVis():
    def __init__(
            self, 
            language, 
            dist_name,
            dists,
            group, 
            distances_dir,
            sentiscores_dir,
            metadata_dir,
            canonscores_dir,
            features_dir):
        self.language = language
        self.dist_name = dist_name
        self.dists=dists
        self.group = group
        self.distances_dir = distances_dir
        self.sentiscores_dir = sentiscores_dir
        self.metadata_dir = metadata_dir
        self.canonscores_dir = canonscores_dir
        self.features_dir = features_dir
        self.mx = dists[self.dist_name]['mx']
        self.file_group_mapping = self._init_colormap()
        self.nmfw = self.dists[self.dist_name]['nmfw']

    def _relabel_axis(self):
        labels = self.ax.get_ymajorticklabels()
        for label in labels:
            color = self.file_group_mapping.loc[self.file_group_mapping['file_name'] ==label.get_text(), 'group_color']
            label = label.set_color(str(color.values[0]))

    def save(self,plt, vis_type):
        plt.savefig(os.path.join(self.distances_dir, f'{vis_type}_{self.dist_name}_{self.group}.png'))

    def draw_dendrogram(self, clustering):
        plt.clf()
        plt.figure(figsize=(12,12),dpi=1000)
        dendro_data = sch.dendrogram(
            Z=clustering, 
            orientation='left', 
            labels=self.mx.index.to_list(),
            show_leaf_counts=True,
            leaf_font_size=1)
        self.ax = plt.gca() 
        self._relabel_axis()
        plt.title(f'{self.dist_name}, {self.group}, {self.language}')
        #plt.xlabel('Samples')
        #plt.ylabel('Euclidean distances')
        self.save(plt, 'dendrogram')

    def draw_mds(self, clustering):
        df = MDS(n_components=2, dissimilarity='precomputed', random_state=8, metric=True).fit_transform(self.mx)
        df = pd.DataFrame(df, columns=['comp1', 'comp2'], index=self.mx.index)
        df = df.merge(self.file_group_mapping, how='inner', left_index=True, right_on='file_name', validate='one_to_one')
        df = df.merge(clustering, how='inner', left_on='file_name', right_index=True, validate='1:1')

        def _group_cluster_color(row):
            color = None
            if row['group_color'] == 'b' and row['cluster'] == 0:
                color = 'darkblue'
            elif row['group_color'] == 'b' and row['cluster'] == 1:
                color = 'royalblue'
            elif row['group_color'] == 'r' and row['cluster'] == 0:
                color = 'crimson'
            #elif row['group_color'] == 'r' and row['cluster'] == 0:
            else:
                color = 'deeppink'
            return color

        df['group_cluster_color'] = df.apply(_group_cluster_color, axis=1)


        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)
        plt.scatter(df['comp1'], df['comp2'], color=df['group_cluster_color'], s=2, label="MDS")
        plt.title(f'{self.dist_name}, {self.group}, {self.language}')
        self.save(plt, 'MDS')

    def _init_colormap(self):
        if self.group == 'author':
            x = get_author_groups(self.mx)
            file_group_mapping = pd.DataFrame(x).reset_index().rename({'index': 'file_name'}, axis=1)
            groups = file_group_mapping['author'].unique()
            props = mpl.rcParams['axes.prop_cycle']
            colormap = {x: y['group_color'] for x,y in zip(groups, props())}
            colormap = pd.DataFrame(colormap, index=['group_color']).T.reset_index().rename({'index': 'author'}, axis=1)
            file_group_mapping = file_group_mapping.merge(colormap, how='left', on='author', validate='many_to_one')

        elif self.group == 'unread':
            X, file_group_mapping = get_data(
                language=self.language, 
                task='regression-importance', 
                label_type='canon', 
                features='book', 
                features_dir=self.features_dir, 
                canonscores_dir=self.canonscores_dir, 
                sentiscores_dir=self.sentiscores_dir, 
                metadata_dir=self.metadata_dir)
            threshold = 0.5
            file_group_mapping = file_group_mapping.reset_index().rename({'index': 'file_name'}, axis=1)
            file_group_mapping['group_color'] = file_group_mapping['y'].apply(lambda x: 'r' if x > threshold else 'b')

        elif self.group == 'gender':
            # Combine author metadata and file_name
            authors = pd.read_csv(os.path.join(self.metadata_dir, 'authors.csv'), header=0, sep=';')[['author_viaf','name', 'first_name', 'gender']]
            metadata = pd.read_csv(os.path.join(self.metadata_dir, f'{self.language.upper()}_texts_meta.csv'), header=0, sep=';')[['author_viaf', 'file_name']]
            file_group_mapping = metadata.merge(authors, how='left', on='author_viaf', validate='many_to_one')
            file_group_mapping['file_name'] = file_group_mapping['file_name'].replace(to_replace={ ############################3
                # new -- old
                'Storm_Theodor_Immensee_1850': 'Storm_Theodor_Immersee_1850',
                'Hoffmansthal_Hugo_Reitergeschichte_1899': 'Hoffmansthal_Hugo-von_Reitergeschichte_1899'
                })
            # check if all file names have metadata
            #ytest = df.merge(self.mx, left_on='file_name', right_index=True, validate='one_to_one', how='outer')
            file_group_mapping['group_color'] = file_group_mapping['gender'].apply(lambda x: 'r' if x=='f' else 'b')
            
        return file_group_mapping





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
        mx.to_csv(
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
        # pydelta.Corpus takes string
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
        print('pydelta dist matrix', mx)
        return mx

def is_symmetric(df):
    return df.equals(df.T)

def show_distance_distribution(mx, language, filename, data_dir):
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
    ax.set_title(f'Distance distribution {filename}')
    plt.xticks(np.arange(0, max(values) + 0.1, step=0.5))
    plt.xticks(rotation=90)

    plt.savefig(os.path.join(data_dir, 'distances', language, f'distance-dist_{filename}.png'))


def get_importances_mx(language, data_dir):
    #----------------------------------------------
    # Get distance based on feature importance
    #----------------------------------------------
    i = ImprtDistance(language, data_dir)
    mx = i.calculate_mx(file_name='imprtdist')
    show_distance_distribution(mx, language, 'imprt', data_dir)
    return mx


# %%
def get_pydelta_mx(language, data_dir, **kwargs):
    #----------------------------------------------
    # Get Pydelta distances
    #----------------------------------------------
    #print(delta.functions)
    nmfw = kwargs['nmfw']
    function = kwargs['function']
    pydelta = PydeltaDist(language, data_dir, nmfw=nmfw)
    #x = corpus.sum(axis=1).sort_values(ascending=False)
    mx = pydelta.calculate_mx(function, file_name=f'pydelta_{function}{nmfw}')
    show_distance_distribution(mx, language, f'{function}{nmfw}', data_dir)
    # print(mx.simple_score())
    return mx
    

def get_mx(distance, language, data_dir, **kwargs):
    mx = None
    if distance == 'imprt':
        mx = get_importances_mx(language, data_dir)
    elif distance == 'burrows500':
        mx = get_pydelta_mx(language, data_dir, **kwargs)
    elif distance == 'burrows1000':
        mx = get_pydelta_mx(language, data_dir, **kwargs)
    return mx

