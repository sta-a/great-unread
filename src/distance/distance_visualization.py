import matplotlib.pyplot as plt
import pandas as pd
import os
import colorcet as cc

import sys
sys.path.append("..")
from utils import DataHandler, DataChecks, get_texts_by_author

import logging
logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
ticker_logger = logging.getLogger('matplotlib.ticker')
ticker_logger.setLevel(logging.WARNING)



class ColorMap(DataHandler):
    '''
    cluster_list: list of filenames or list of lists where each inner list represents one cluster
    '''
    def __init__(
            self,
            attribute_name, 
            cluster_list,
            language = None):
        super().__init__(language, output_dir=None)        
        self.attribute_name = attribute_name
        self.cluster_list = cluster_list

    @staticmethod
    def get_colors_discrete():
        '''
        Get discrete colors that are maximally (greedy) different from previous ones 
        '''
        colors = iter(cc.glasbey_hv) #hsv
        return colors

    def get_colors_sequential(self, val):
        '''
        val: Number between 0 and 1
        '''
        cmap = plt.cm.get_cmap('gist_yarg')
        color = cmap(val)
        color = self.color_for_pygraphviz(color)
        return color
    
    def color_for_pygraphviz(self, color):
        # HSV needs to be passed as a string for pygraphviz
        return ' '.join(str(i) for i in color)

    def map_colors_authors(self):
        '''
        Map each filename to a color so that texts by the same author have the same color.
        '''

        # Map each author to a color.
        author_filename_mapping, _ = get_texts_by_author(self.cluster_list)
        colors = ColorMap.get_colors_discrete()
        author_color_mapping = {author: self.color_for_pygraphviz(next(colors)) for author, _ in author_filename_mapping.items()}

        fn_color_mapping = {}
        for author, self.cluster_list in author_filename_mapping.items():
            for fn in self.cluster_list:
                fn_color_mapping[fn] = author_color_mapping[author]
        return fn_color_mapping

    def get_metadata(self):
        if self.attribute_name == 'canon':
            df = DataChecks(self.language).prepare_metadata(data=self.attribute_name)
            df = df.rename(columns={'m3': 'score'})
            df = df.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
            df = df.drop_duplicates(subset='file_name')
        # discretize
        # file_group_mapping['scores'].apply(lambda x: 'r' if x > threshold else 'b')

        elif self.attribute_name == 'gender':
            df = DataChecks(self.language).prepare_metadata(data=self.attribute_name)
        return df


    def get_color_map(self):
        fn_color_mapping = {}
        if (self.attribute_name == 'author') or (self.attribute_name == 'canon') or (self.attribute_name == 'gender'):
            
            if not all(isinstance(item, str) for item in self.cluster_list):
                raise ValueError('Pass list of strings if attribute_name is \'author\' or \'unread\'.')
            
            if self.attribute_name == 'author':
                fn_color_mapping = self.map_colors_authors()
            
            elif self.attribute_name == 'canon':
                scores = self.get_metadata()
                scores = scores.loc[scores['file_name'].isin(self.cluster_list)]
                fn_color_mapping = {row['file_name']: self.get_colors_sequential(row['score']) for _, row in scores.iterrows()} ###################33mapping everywheere the same
            
            elif self.attribute_name == 'gender':
                gender = self.get_metadata()
                gender_color_map = {'f': 'red', 'm': 'blue', 'b': 'yellow', 'a': 'lightgreen'}
                gender['color'] = gender['gender'].map(gender_color_map)
                fn_color_mapping = {row['file_name']: row['color'] for _, row in gender.iterrows()}
        
        else:
            if not all(isinstance(item, list) for item in self.cluster_list):
                raise ValueError('Pass list of lists if attribute_name is a clustering algorithm.')
            colors = ColorMap.get_colors_discrete()
            for cluster in self.cluster_list:
                color = next(colors)
                for fn in cluster:
                    fn_color_mapping[fn] = color
        return fn_color_mapping

