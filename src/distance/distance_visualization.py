import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import os
import colorcet as cc
import networkx as nx
import sys
sys.path.append("..")
print(sys.path)
from utils import get_texts_by_author
from .distance_analysis import check_symmetric


class ColorMap():
    '''
    cluster_list: list of filenames or list of lists where each inner list represents one cluster
    '''
    def __init__(
            self,
            cluster_name, 
            cluster_list,
            metadata_dir, 
            canonscores_dir,
            language = None):
        
        self.cluster_name = cluster_name
        self.cluster_list = cluster_list
        self.metadata_dir = metadata_dir
        self.canonscores_dir = canonscores_dir
        self.language = language

    @staticmethod
    def get_colors_discrete():
        '''
        Get discrete colors that are maximally (greedy) different from previous ones 
        '''
        colors = iter(cc.glasbey_hv) #hsv
        return colors

    @staticmethod
    def get_colors_sequential(val):
        '''
        val: Number between 0 and 1
        '''
        cmap = plt.cm.get_cmap('gist_yarg')
        color = cmap(val)
        return color

    def map_colors_authors(self):
        '''
        Map each filename to a color so that texts by the same author have the same color.
        '''

        # Map each author to a color.
        author_filename_mapping, _ = get_texts_by_author(self.cluster_list)
        colors = ColorMap.get_colors_discrete()
        author_color_mapping = {author: next(colors) for author, _ in author_filename_mapping.items()}

        fn_color_mapping = {}
        for author, self.cluster_list in author_filename_mapping.items():
            for fn in self.cluster_list:
                fn_color_mapping[fn] = author_color_mapping[author]
        return fn_color_mapping

    def get_canonscores(self):
        canon_file = '210907_regression_predict_02_setp3_FINAL.csv'
        scores = pd.read_csv(os.path.join(self.canonscores_dir, canon_file), sep=';')[['file_name', 'm3']]
        scores = scores.rename(columns={'m3': 'score'})
        scores = scores.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
        scores = scores.drop_duplicates(subset='file_name')

        # discretize
        # file_group_mapping['scores'].apply(lambda x: 'r' if x > threshold else 'b')
        return scores
    
    def get_gender(self):
        # Combine author metadata and file_name
        authors = pd.read_csv(os.path.join(self.metadata_dir, 'authors.csv'), header=0, sep=';')[['author_viaf','name', 'first_name', 'gender']]
        metadata = pd.read_csv(os.path.join(self.metadata_dir, f'{self.language.upper()}_texts_meta.csv'), header=0, sep=';')[['author_viaf', 'file_name']]
        gender = metadata.merge(authors, how='left', on='author_viaf', validate='many_to_one')
        gender['file_name'] = gender['file_name'].replace(to_replace={ #######################################
            # new -- old
            'Storm_Theodor_Immensee_1850': 'Storm_Theodor_Immersee_1850',
            'Hoffmansthal_Hugo_Reitergeschichte_1899': 'Hoffmansthal_Hugo-von_Reitergeschichte_1899'
            })
        gender['gender'] = gender['gender'].replace(to_replace={'w':'f'})#################################3
        # NA
        # Missing values
        if self.language == 'ger':
            gender.loc['Hebel_Johann-Peter_Kannitverstan_1808', 'gender'] = 'm'
            gender.loc['May_Karl_Ardistan-und-Dschinnistan_1909', 'gender'] = 'm'
            gender.loc['May_Karl_Das-Waldroeschen_1883', 'gender'] = 'm'

        return gender

    def get_color_map(self):
        fn_color_mapping = {}
        if (self.cluster_name == 'author') or (self.cluster_name == 'unread') or (self.cluster_name == 'gender'):
            if not all(isinstance(item, str) for item in self.cluster_list):
                raise ValueError('Pass list of strings if cluster_name is \'author\' or \'unread\'.')
            if self.cluster_name == 'author':
                fn_color_mapping = self.map_colors_authors()
            elif self.cluster_name == 'unread':
                scores = self.get_canonscores()
                scores = scores.loc[scores['file_name'].isin(self.cluster_list)]
                fn_color_mapping = {row['file_name']: ColorMap.get_colors_sequential(row['score']) for _, row in scores.iterrows()} ###################33mapping everywheere the same
            elif self.cluster_name == 'gender':
                gender = self.get_gender()
                gender['color'] = gender['gender'].apply(lambda x: 'r' if x=='f' else ('b' if x == 'm' else 'green')) # anonymous/NaN is green
                fn_color_mapping = {row['file_name']: row['color'] for _, row in gender.iterrows()}
        else:
            if not all(isinstance(item, list) for item in self.cluster_list):
                raise ValueError('Pass list of lists if cluster_name is a clustering algorithm.')
            colors = ColorMap.get_colors_discrete()
            for cluster in self.cluster_list:
                color = next(colors)
                for fn in cluster:
                    fn_color_mapping[fn] = color
        return fn_color_mapping


class NetworkViz():
    def __init__(
            self, 
            mx = None, 
            G = None,
            draw = True,
            cluster_name = None, # cluster: unlabeled groups
            attribute_name = None, # attribute: labeled groups, i.e. 'm', 'f' #################################
            distances_dir = None,
            metadata_dir = None,
            canonscores_dir = None,
            language = None):

        self.mx = mx
        self.G = G
        self.draw = draw,
        self.cluster_name = cluster_name
        self.attribute_name = attribute_name, 
        self.distances_dir = distances_dir
        self.metadata_dir = metadata_dir
        self.canonscores_dir = canonscores_dir
        self.language = language

        if (self.mx is not None and self.G is not None) or (self.mx is None and self.G is None):
            raise ValueError('Pass either matrix or graph.')
        if self.G is None:
            self.G = self.nx_graph_from_mx()
        if self.cluster_name is not None:
            self.clusters = self.get_communities()
        if self.draw:
            self.draw_graph()

    def nx_graph_from_mx(self):
        '''
        mx: Adjacency or directed or undirected weight matrix
        '''
        mx = self.mx.fillna(0) # Adjacency matrix must not contain Nan!
        if check_symmetric(mx):
            G = nx.from_pandas_adjacency(mx)
            print('Matrix is symmetric.')
        else:
            G = nx.from_pandas_adjacency(mx, create_using=nx.DiGraph) 
            print('Matrix is not symmetric but directed.')
        G.remove_edges_from(nx.selfloop_edges(G))
        return G

    def get_communities(self):
        if self.cluster_name == 'louvain':
            c = nx.community.louvain_communities(self.G, weight='weight', seed=11, resolution=0.1)
            return c
    
    def draw_graph(self):
        if self.cluster_name is not None:
            if (self.cluster_name == 'author') or (self.cluster_name == 'unread') or (self.cluster_name == 'gender'):
                cluster_list = list(self.G.nodes(data=False)) # get list of file names
            else:
                cluster_list = self.clusters

            fn_color_mapping = ColorMap(
                self.cluster_name,
                cluster_list,
                self.metadata_dir, 
                self.canonscores_dir,
                self.language).get_color_map()
            assert self.G.number_of_nodes() == len(fn_color_mapping)
            color_map = [fn_color_mapping[node] for node in self.G]
        else:
            color_map = 'blue'

        pos = nx.spring_layout(self.G, seed=8)
        nx.draw_networkx(self.G, pos, node_size=30, with_labels=False, node_color=color_map)
        plt.show()





