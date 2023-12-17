# %%

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from typing import List
import colorcet as cc
import itertools
import sys
sys.path.append("..")
from sklearn.preprocessing import minmax_scale
from copy import deepcopy
from matplotlib import markers
from utils import DataHandler, DataLoader, TextsByAuthor

import logging
logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
# ticker_logger = logging.getLogger('matplotlib.ticker')
# ticker_logger.setLevel(logging.WARNING)

class MetadataHandler(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='similarity', data_type='png')


    def get_metadata(self, add_color=True):
        gender = DataLoader(self.language).prepare_metadata(type='gender')
        gender = gender[['gender']]
        gender['gender'] = gender['gender'].map({'m': 0, 'f': 1, 'a': 2, 'b': 3})

        canon = DataLoader(self.language).prepare_metadata(type='canon')

        features = DataLoader(self.language).prepare_features()

        author_filename_mapping = TextsByAuthor(self.language).author_filename_mapping
        author = pd.DataFrame([(author, work) for author, works in author_filename_mapping.items() for work in works],
                        columns=['author', 'file_name'])
        author = author.set_index('file_name', inplace=False)

        fn = [os.path.splitext(f)[0] for f in os.listdir(self.text_raw_dir) if f.endswith('.txt')]
        year = pd.DataFrame({'file_name': fn})
        year['year'] = year['file_name'].str[-4:].astype(int)
        year = year.set_index('file_name', inplace=False)


        # Merge dataframes based on their indices
        df_list = [gender, author, year, canon, features]
        metadf = pd.DataFrame()
        for df in df_list:
            if metadf.empty:
                metadf = df
            else:
                metadf = pd.merge(metadf, df, left_index=True, right_index=True, how='inner', validate='1:1')

        assert len(metadf) == self.nr_texts

        if add_color:
            cm = ColorMap(metadf)
            for col in metadf.columns:
                cm.add_color_column(col)
            metadf = cm.metadf
        return metadf

    
    
    def add_color_to_df(self, metadf, colname):
        # Add a color column to a df
        self.cm = ColorMap(metadf)
        metadf = self.cm.add_color_column(colname)
        metadf = self.cm.metadf

        assert len(metadf) == self.nr_texts
        return metadf
    
    
    def add_shape_to_df(self, metadf):
        # Add a shape column for the cluster column
        self.cm = ShapeMap(metadf)
        metadf = self.cm.add_shape_column()
        metadf = self.cm.metadf

        assert len(metadf) == self.nr_texts
        return metadf


class ShapeMap():
    '''
    Match clusters to shapes.
    Only implemented for 'cluster' column, use colors for other attributes.
    '''
    # Available shapes in nx
    SHAPES = ['o', 'X', '*', 'd', 'P', 'p', '<', 'H', '>', 's', 'v', 'h', '8', 'D', '^']
    
    def __init__(self, metadf):
        self.metadf = metadf
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def add_shape_column(self):
        self.map_list()

    def map_list(self):
        # Map each cluster to a shape. 
        # Cluster with label 0 is mapped to the first list element, label 1 to the second element, and so on.
        # If end of the list is reached, start from the beginning of the list.

        # Map clusters to indices in the SHAPES list
        self.metadf[f'clst_shape'] = self.metadf['cluster'] % len(self.SHAPES)
        # Map indices to shapes
        self.metadf[f'clst_shape'] = self.metadf[f'clst_shape'].apply(lambda x: self.SHAPES[x])
        # if self.metadf['cluster'].nunique() > len(self.SHAPES):
        #     self.logger.debug(f'Number of unique elements in "cluster" exceeds the number of shapes in SHAPES list. Different clusters have the same shape.')


class Colors():
    def __init__(self):
        # Lut: the number of colors that are generated from the color map
        # Set to a high value to get a different color for values that are close together
        self.cmap = plt.cm.get_cmap('seismic', lut=10000) # gist_yarg

    @staticmethod
    def get_colors_discrete():
        '''
        Get discrete colors that are maximally (greedy) different from previous ones 
        The 'glasbey_hv' colormap contains 256 unique colors.
        '''
        colors = iter(cc.glasbey_bw_minc_20) # color palette with no greys
        return colors

    def get_colors_sequential(self, val):
        '''
        val: Number between 0 and 1
        '''
        if pd.isna(val):
            return 'green'
        color = self.cmap(val)
        return color
    
    def color_for_pygraphviz(self, color):
        # HSV needs to be passed as a string for pygraphviz
        return ' '.join(str(i) for i in color)


class ColorMap(Colors):
    '''
    Map a column of metadf to a color.
    The columns is either a metadata attribute (gender, author etc.), or the cluster assingments.
    '''
    def __init__(self, metadf, pgv=False):
        super().__init__()
        self.metadf = metadf
        self.pgv = pgv


    def add_color_column(self, colname):
        if colname == 'gender':
            self.map_gender(colname)
        elif (colname == 'cluster') or (colname == 'author'):
            self.map_categorical(colname)
        else:
            self.map_continuous(colname)

        # Transform format so it is compatible with pgv
        if self.pgv:
            self.metadf[f'{colname}_color'] = self.metadf[f'{colname}_color'].apply(self.color_for_pygraphviz)


    def map_gender(self, colname):
        gender_color_map = {0: 'blue', 1: 'red', 2: 'lightgreen', 3: 'yellow'}
        self.metadf[f'{colname}_color'] = self.metadf['gender'].map(gender_color_map)
    
    # def map_categorical(self, colname):
    #     self.metadf[f'{colname}_color'] = self.metadf[colname].map(dict(zip(sorted(self.metadf[colname].unique()), self.get_colors_discrete())))

    
    # def map_categorical(self, colname):
    #     value_counts = self.metadf[colname].value_counts()

    #     # Get colors for unique elements
    #     unique_elements = sorted(self.metadf[colname].unique())
    #     colors = self.get_colors_discrete()

    #     # Create a mapping with black color for elements occurring only once
    #     color_mapping = {element: color if count > 1 else 'black' for element, color, count in zip(unique_elements, colors, value_counts)}

    #     self.metadf[f'{colname}_color'] = self.metadf[colname].map(color_mapping)


    def map_categorical(self, colname):
        # Count the occurrences of each unique value in the specified column
        value_counts = self.metadf[colname].value_counts()

        # Get a sorted list of unique elements in the column
        unique_elements = sorted(self.metadf[colname].unique())
        
        # Create an iterator that cycles through colors
        colors = self.get_colors_discrete()
        # colors = itertools.cycle(colors)


        # Create a mapping with cycling through colors for all unique elements
        color_mapping = {
            element: next(colors) if count > 1 else 'darkgrey'
            for element, count in zip(unique_elements, value_counts)
        }

        # Create a new column with the original column name appended with '_color'
        self.metadf[f'{colname}_color'] = self.metadf[colname].map(color_mapping)


    def map_continuous(self, colname):
        # Scale values so that lowest value is 0 and highest value is 1
        scaled_col = pd.Series(minmax_scale(self.metadf[colname]))
        color_col = scaled_col.apply(self.get_colors_sequential)
        self.metadf = self.metadf.assign(newcol=color_col.values).rename(columns={'newcol': f'{colname}_color'})



class CombinationInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.omit_default = ['metadf', 'param_comb', 'spars_param', 'omit_default', 'cluster_alg']
        self.paramcomb_to_string()       
        self.spars_to_string()
        # self.extra = '' # Store additional information


    def spars_to_string(self):
        if hasattr(self, 'sparsmode') and hasattr(self, 'spars_param'):
            if self.spars_param is None:
                self.sparsmode = self.sparsmode
            elif isinstance(self.spars_param, tuple):
                self.sparsmode = f'{self.sparsmode}-{self.spars_param[0]}-{self.spars_param[1]}'
            else:
                self.sparsmode = f'{self.sparsmode}-{str(self.replace_dot(self.spars_param))}'



    def paramcomb_to_string(self):
        if hasattr(self, 'param_comb'):
            self.clst_alg_params = None
            if bool(self.param_comb):
                paramstr = '-'.join([f'{key}-{self.replace_dot(value)}' for key, value in self.param_comb.items()])
                self.clst_alg_params = f'{self.cluster_alg}-{paramstr}'


    def replace_dot(self, value):
        # Use '%' to mark dot in float
        if isinstance(value, float):
            return str(value).replace('.', '%')
        return value


    def as_string(self, sep='_', omit: List[str] = []):
        omit_lst = self.omit_default + omit # + ['extra']

        filtered_values = []
        for key, value in self.__dict__.items():
            if key not in omit_lst:
                if value is not None and (not isinstance(value, dict)):
                    filtered_values.append(str(self.replace_dot(value)))

        return sep.join(filtered_values)
    

    def as_df(self, omit: List[str] = []):
        omit_lst = self.omit_default + omit
        data = {key: value for key, value in self.__dict__.items() if key not in omit_lst}
        return pd.DataFrame([data], index=[0])


    def as_dict(self):
        return self.__dict__
    


# %%
