# %%

import matplotlib.pyplot as plt
import pandas as pd
import os
import colorcet as cc
import itertools
import sys
sys.path.append("..")
from utils import DataHandler, DataLoader, TextsByAuthor
from copy import deepcopy

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
            attr=None,
            ):
        super().__init__(language, output_dir='similarity', data_type='png')
        self.attr = attr
        # self.metadf = self.get_metadata()


    def get_metadata(self):
        if self.attr == 'canon':
            df = DataLoader(self.language).prepare_metadata(type=self.attr)
        # discretize
        # file_group_mapping['scores'].apply(lambda x: 'r' if x > threshold else 'b')

        elif self.attr == 'gender':
            df = DataLoader(self.language).prepare_metadata(type=self.attr)

        elif self.attr == 'features':
            df = DataLoader(self.language).prepare_features() # file name is index

        elif self.attr == 'author':
            author_filename_mapping = TextsByAuthor(self.language).author_filename_mapping
            df = pd.DataFrame([(author, work) for author, works in author_filename_mapping.items() for work in works],
                            columns=['author', 'file_name'])
            df = df.set_index('file_name', inplace=False)

        elif self.attr == 'year':
            fn = [os.path.splitext(f)[0] for f in os.listdir(self.text_raw_dir) if f.endswith('.txt')]
            df = pd.DataFrame({'file_name': fn})
            df['year'] = df['file_name'].str[-4:].astype(int)
            df = df.set_index('file_name', inplace=False)
        
        return df
    

class Colors():
    def __init__(self):
        pass

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
        # color = self.color_for_pygraphviz(color) #################################
        return color
    
    def color_for_pygraphviz(self, color):
        # HSV needs to be passed as a string for pygraphviz
        return ' '.join(str(i) for i in color)


class ColorMap(Colors):
    '''
    Map the column with name 'attr' of metadf to a color.
    'attr' is either a metadata attribute (gender, author etc.), or the cluster assingments.
    '''
    def __init__(self, metadf, attr='cluster'):
        self.metadf = metadf
        self.attr = attr

    def get_color_map(self, pgv=False):
        if self.attr == 'author':
            self.map_author()
        elif self.attr == 'canon':
            self.map_canon()
        elif self.attr == 'gender':
            self.map_gender()
        elif self.attr == 'cluster':
            self.map_cluster()

        # Transform format so it is compatible with pgv
        if pgv:
            self.metadf['color'] = self.metadf['color'].apply(self.color_for_pygraphviz)

        return self.metadf

    def map_cluster(self):
        self.metadf['color'] = self.metadf['cluster'].map(dict(zip(sorted(self.metadf['cluster'].unique()), self.get_colors_discrete())))


    def map_author(self):
        '''
        Map each filename to a color so that texts by the same author have the same color.
        '''
        colors = ColorMap.get_colors_discrete()
        author_color_mapping = {author: next(colors) for author in self.metadfdf['author'].unique()}
        self.metadf['color'] = self.metadf['author'].map(author_color_mapping)


    def map_canon(self):
        #self.metadf = self.self.metadf.loc[self.metadf['file_name'].isin(self.clusters)]
        fn_color_mapping = {row['file_name']: self.get_colors_sequential(row['canon']) for _, row in self.metadf.iterrows()}
        return fn_color_mapping
    
    def map_gender(self):
        gender_color_map = {'f': 'red', 'm': 'blue', 'b': 'yellow', 'a': 'lightgreen'}
        self.metadf['color'] = self.metadf['gender'].map(gender_color_map)
        fn_color_mapping = {row['file_name']: row['color'] for _, row in self.metadf.iterrows()}
        return fn_color_mapping
    




# class ShapeMap(MetadataHandler):
#     '''
#     If attr is set, map filenames to shape according to attribute.
#     If clusters is set, map each cluster to a shape.
#     '''
#     def __init__(
#             self,
#             language=None,
#             attr=None,
#             clusters=None):
#         super().__init__( language=language)
#         self.attr = attr
#         self.clusters = clusters
#         self.shapes = itertools.cycle(['egg', 'invtrapezium', 'star', 'cylinder', 'cds', 'terminator', 'box', 'tripleoctagon','Mdiamond', 'Mcircle', 'Mdiamond', 'Mcircle', 'Msquare', 'circle', 'diamond', 'doublecircle', 'doubleoctagon', 'ellipse', 'hexagon', 'house', 'invhouse', 'invtriangle', 'none', 'octagon', 'oval', 'parallelogram', 'pentagon', 'plaintext', 'point', 'polygon', 'rectangle', 'septagon', 'square', 'tab', 'trapezium', 'triangle'])

#         if self.attr is not None and self.clusters is not None:
#             raise ValueError('Pass either attribute name of cluster list.')

#     def color_for_pygraphviz(self, color):
#         # HSV needs to be passed as a string for pygraphviz
#         return ' '.join(str(i) for i in color)
    
#     def map_author(self):
#         '''
#         Map each filename to a shape so that texts by the same author have the same shape.
#         '''
#         # author_filename_mapping: dict{author name: [list with all works by author]}
#         author_filename_mapping = TextsByAuthor().author_filename_mapping
#         author_shape_mapping = {author: next(self.shapes) for author, _ in author_filename_mapping.items()}

#         fn_shape_mapping = {fn: author_shape_mapping[author] for author, filenames in author_filename_mapping.items() for fn in filenames}

#         # fn_shape_mapping = {}
#         # for author, list_of_filenames in author_filename_mapping.items():
#         #     for fn in list_of_filenames:
#         #         fn_shape_mapping[fn] = author_shape_mapping[author]
#         return fn_shape_mapping

#     # def map_canon(self):
#     #     #self.metadf = self.self.metadf.loc[self.metadf['file_name'].isin(self.clusters)]
#     #     fn_shape_mapping = {row['file_name']: self.get_shapes_sequential(row['canon']) for _, row in self.metadf.iterrows()}
#     #     return fn_shape_mapping
    
#     def map_gender(self):
#         gender_shape_map = {'f': 'circle', 'm': 'triangle', 'b': 'star', 'a': 'hexagon'}
#         self.metadf['shape'] = self.metadf['gender'].map(gender_shape_map)
#         fn_shape_mapping = {row['file_name']: row['shape'] for _, row in self.metadf.iterrows()}
#         return fn_shape_mapping
    

#     def get_shape_map(self):
#         if self.attr is not None:
#             fn_shape_mapping = self.match_attr_value()
#         else:
#             fn_shape_mapping = self.map_cluster()
#         return fn_shape_mapping
    

#     def map_cluster(self):
#         fn_shape_mapping = {}
#         # if not all(isinstance(item, list) for item in self.clusters):
#         #     raise ValueError('Not a valid clustering.') ######3
        
#         for cluster in self.clusters:
#             shape = next(self.shapes)
#             for fn in cluster:
#                 fn_shape_mapping[fn] = shape
#         return fn_shape_mapping
    
    
# class CombinationInfo:
#     def __init__(self, mxname, cluster_alg, attr, param_comb):
#         self.mxname = mxname
#         self.cluster_alg = cluster_alg
#         self.attr = attr
#         self.param_comb = self.paramcomb_to_string(param_comb)

#     def paramcomb_to_string(self, param_comb):
#         return'-'.join([f'{key}-{value}' for key, value in param_comb.items()])

#     def as_string(self):
#         return f'{self.mxname}_{self.cluster_alg}_{self.attr}_{self.param_comb}'

#     def as_df(self):
#         # data = {key: [getattr(self, key)] for key in ['mxname', 'cluster_alg', 'attr']}
#         # data.update(self.param_comb)

#         return pd.DataFrame([vars(self)], index=[0])
#         # print(df)
#         # param_comb = self.paramcomb_to_string()
#         # df['param_comb'] = param_comb

#     def as_dict(self):
#         return {'mxname': self.mxname, 'cluster_alg': self.cluster_alg, 'attr': self.attr, 'param_comb': self.param_comb}

class CombinationInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.param_comb = self.paramcomb_to_string(self.param_comb)

    def paramcomb_to_string(self, param_comb):
        return '-'.join([f'{key}-{value}' for key, value in param_comb.items()])

    def as_string(self):
        return '_'.join(str(value) for value in self.__dict__.values())

    def as_df(self):
        # data = {key: [value] for key, value in self.__dict__.items()}
        # return pd.DataFrame([data], index=[0])
        return pd.DataFrame([vars(self)], index=[0])

    def as_dict(self):
        return self.__dict__