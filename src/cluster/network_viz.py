import matplotlib.pyplot as plt
import pandas as pd
import os
import colorcet as cc
import itertools
import sys
sys.path.append("..")
from utils import DataHandler, DataChecks, TextsByAuthor

import logging
logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
ticker_logger = logging.getLogger('matplotlib.ticker')
ticker_logger.setLevel(logging.WARNING)

class MetadataHandler(DataHandler):
    def __init__(
            self,
            language = None,
            attribute_name=None,
            ):
        super().__init__(language, output_dir=None)
        self.attribute_name = attribute_name


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
        else:
            df = None
        return df
    
    def match_attr_value(self):
        if self.attribute_name == 'author':
            mapping = self.map_author()
        
        elif self.attribute_name == 'canon':
            mapping = self.map_canon()
        
        elif self.attribute_name == 'gender':
            mapping = self.map_gender()
        return mapping




class ColorMap(MetadataHandler):
    '''
    If attribute_name is set, map filenames to color according to attribute.
    If cluster_list is set, map each cluster to a color.
    '''
    def __init__(
            self,
            language=None,
            attribute_name=None, 
            cluster_list=None):
        super().__init__(attribute_name=attribute_name, language=language)
        self.cluster_list = cluster_list
        self.metadf = self.get_metadata()
        if self.attribute_name is not None and self.cluster_list is not None:
            raise ValueError('Pass either attribute name of cluster list.')

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

    def map_author(self):
        '''
        Map each filename to a color so that texts by the same author have the same color.
        '''
        # author_filename_mapping: dict{author name: [list with all works by author]}
        author_filename_mapping = TextsByAuthor().author_filename_mapping
        colors = ColorMap.get_colors_discrete()
        author_color_mapping = {author: self.color_for_pygraphviz(next(colors)) for author, _ in author_filename_mapping.items()}

        fn_color_mapping = {fn: author_color_mapping[author] for author, filenames in author_filename_mapping.items() for fn in filenames}

        # fn_color_mapping = {}
        # for author, list_of_filenames in author_filename_mapping.items():
        #     for fn in list_of_filenames:
        #         fn_color_mapping[fn] = author_color_mapping[author]
        return fn_color_mapping

    def map_canon(self):
        #self.metadf = self.self.metadf.loc[self.metadf['file_name'].isin(self.cluster_list)]
        fn_color_mapping = {row['file_name']: self.get_colors_sequential(row['score']) for _, row in self.metadf.iterrows()}
        return fn_color_mapping
    
    def map_gender(self):
        gender_color_map = {'f': 'red', 'm': 'blue', 'b': 'yellow', 'a': 'lightgreen'}
        self.metadf['color'] = self.metadf['gender'].map(gender_color_map)
        fn_color_mapping = {row['file_name']: row['color'] for _, row in self.metadf.iterrows()}
        return fn_color_mapping
    
    def map_cluster(self):
        fn_color_mapping = {}
        if not all(isinstance(item, list) for item in self.cluster_list):
            raise ValueError('Not a valid clustering.')
        
        colors = ColorMap.get_colors_discrete()
        print(len(self.cluster_list))
        for cluster in self.cluster_list:
            color = self.color_for_pygraphviz(next(colors))
            for fn in cluster:
                fn_color_mapping[fn] = color
        return fn_color_mapping

    def get_color_map(self):
        if self.attribute_name is not None:
            fn_color_mapping = self.match_attr_value()
        elif self.cluster_list is not None:
            fn_color_mapping = self.map_cluster()
        return fn_color_mapping



class ShapeMap(MetadataHandler):
    '''
    If attribute_name is set, map filenames to shape according to attribute.
    If cluster_list is set, map each cluster to a shape.
    '''
    def __init__(
            self,
            language=None,
            attribute_name=None,
            cluster_list=None):
        super().__init__( language=language)
        self.attribute_name = attribute_name
        self.cluster_list = cluster_list
        self.metadf = self.get_metadata()
        self.shapes = itertools.cycle(['egg', 'invtrapezium', 'star', 'cylinder', 'cds', 'terminator', 'box', 'tripleoctagon','Mdiamond', 'Mcircle', 'Mdiamond', 'Mcircle', 'Msquare', 'circle', 'diamond', 'doublecircle', 'doubleoctagon', 'ellipse', 'hexagon', 'house', 'invhouse', 'invtriangle', 'none', 'octagon', 'oval', 'parallelogram', 'pentagon', 'plaintext', 'point', 'polygon', 'rectangle', 'septagon', 'square', 'tab', 'trapezium', 'triangle'])

        if self.attribute_name is not None and self.cluster_list is not None:
            raise ValueError('Pass either attribute name of cluster list.')

    def color_for_pygraphviz(self, color):
        # HSV needs to be passed as a string for pygraphviz
        return ' '.join(str(i) for i in color)
    
    def map_author(self):
        '''
        Map each filename to a shape so that texts by the same author have the same shape.
        '''
        # author_filename_mapping: dict{author name: [list with all works by author]}
        author_filename_mapping = TextsByAuthor().author_filename_mapping
        author_shape_mapping = {author: next(self.shapes) for author, _ in author_filename_mapping.items()}

        fn_shape_mapping = {fn: author_shape_mapping[author] for author, filenames in author_filename_mapping.items() for fn in filenames}

        # fn_shape_mapping = {}
        # for author, list_of_filenames in author_filename_mapping.items():
        #     for fn in list_of_filenames:
        #         fn_shape_mapping[fn] = author_shape_mapping[author]
        return fn_shape_mapping

    # def map_canon(self):
    #     #self.metadf = self.self.metadf.loc[self.metadf['file_name'].isin(self.cluster_list)]
    #     fn_shape_mapping = {row['file_name']: self.get_shapes_sequential(row['score']) for _, row in self.metadf.iterrows()}
    #     return fn_shape_mapping
    
    def map_gender(self):
        gender_shape_map = {'f': 'circle', 'm': 'triangle', 'b': 'star', 'a': 'hexagon'}
        self.metadf['shape'] = self.metadf['gender'].map(gender_shape_map)
        fn_shape_mapping = {row['file_name']: row['shape'] for _, row in self.metadf.iterrows()}
        return fn_shape_mapping
    

    def get_shape_map(self):
        if self.attribute_name is not None:
            fn_shape_mapping = self.match_attr_value()
        else:
            fn_shape_mapping = self.map_cluster()
        return fn_shape_mapping
    

    def map_cluster(self):
        fn_shape_mapping = {}
        # if not all(isinstance(item, list) for item in self.cluster_list):
        #     raise ValueError('Not a valid clustering.') ######3
        
        for cluster in self.cluster_list:
            shape = next(self.shapes)
            for fn in cluster:
                fn_shape_mapping[fn] = shape
        return fn_shape_mapping