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
from matplotlib.colors import to_rgba
from scipy.stats import skew

from utils import DataHandler, DataLoader, TextsByAuthor, FeaturesLoader

import logging
logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
# ticker_logger = logging.getLogger('matplotlib.ticker')
# ticker_logger.setLevel(logging.WARNING)



class MetadataHandler(DataHandler):
    def __init__(self, language, by_author=False):
        super().__init__(language, output_dir=None, data_type='png', by_author=by_author)
        self.by_author = by_author
        self.cat_attrs = ['gender', 'author', 'canon-ascat', 'year-ascat']


    def add_title_mapping(self, tmap, df):
        df = df.merge(tmap, left_index=True, right_on='file_name', validate='1:1')
        df = df.drop(columns=['author', 'file_name'])
        return df


    def get_metadata(self, add_color=False):
        gender = DataLoader(self.language).prepare_metadata(type='gender')
        gender = gender[['gender']]
        gender['gender'] = gender['gender'].map({'m': 0, 'f': 1, 'a': 2, 'b': 3})

        canon = DataLoader(self.language).prepare_metadata(type='canon')
        canon['canon-ascat'] = canon['canon'].apply(lambda x: 0 if x < 0.333 else 1 if x < 0.666 else 2)


        if self.by_author:
            output_dir = self.create_output_dir(output_dir='title_mapping')
            tmap = pd.read_csv(os.path.join(output_dir, 'title_mapping.csv'), header=0)

            gender = self.add_title_mapping(tmap, gender)
            # 'Stevenson-Grift_Robert-Louis-Fanny-van-de_The-Dynamiter_1885' has gender 'b'
            # Text is treated as one of Stevenson's texts
            gender.loc[gender['new_file_name'] == 'Stevenson_Robert-Louis_all_1888', 'gender'] = 0
            assert gender.groupby('new_file_name')['gender'].nunique().max() == 1, 'Each author should have only one value in the gender column.'
            gender = gender.drop_duplicates(subset=['new_file_name'])
            gender = gender.set_index('new_file_name')
            
            # Average over canon scores of individual texts
            canon = self.add_title_mapping(tmap, canon)
            canon_ascat_col = canon['canon-ascat']
            canon = canon.groupby('new_file_name')['canon'].agg(['mean', 'max', 'min'])
            canon.columns = ['canon', 'canon-max', 'canon-min']
            canon['canon-ascat'] = canon['canon'].apply(lambda x: 0 if x < 0.333 else 1 if x < 0.666 else 2)

        # Load features scaled to range between 0 and 1
        features = FeaturesLoader(self.language, by_author=self.by_author).prepare_features(scale=True)
        # Check if all feature values are between 0 and 1 (except for nan)
        features_test = features.fillna(0)
        assert ((features_test >= 0) & (features_test <= 1) | (features_test.isin([0, 1]))).all().all()

        author_filename_mapping = TextsByAuthor(self.language, by_author=self.by_author).author_filename_mapping
        author = pd.DataFrame([(author, work) for author, works in author_filename_mapping.items() for work in works],
                        columns=['author', 'file_name'])
        author = author.set_index('file_name', inplace=False)

        fn = [os.path.splitext(f)[0] for f in os.listdir(self.text_raw_dir) if f.endswith('.txt')]
        year = pd.DataFrame({'file_name': fn})
        year['year'] = year['file_name'].str[-4:].astype(int)
        year = year.set_index('file_name', inplace=False)
        year['year-ascat'] = year['year'].apply(lambda x: 0 if x < 1800 else 1 if x < 1850 else 2 if x < 1900 else 3)


        # Merge dataframes based on their indices
        df_list = [gender, author, year, canon, features]

        metadf = pd.DataFrame()
        for df in df_list:
            if metadf.empty:
                metadf = df
            else:
                metadf = pd.merge(metadf, df, left_index=True, right_index=True, how='inner', validate='1:1')

        # Remove '_full' from column names
        metadf = metadf.rename(columns=lambda x: x.rstrip('_full'))
        # Replace '_' in feature names with '-'
        metadf = metadf.rename(columns=lambda x: x.replace('_', '-'))

        if self.by_author:
            nr_authors = sum(1 for _ in os.listdir(self.text_raw_dir) if os.path.isfile(os.path.join(self.text_raw_dir, _)))
            assert nr_authors == len(metadf)

        else:
            assert len(metadf) == self.nr_texts

        if add_color:
            metadf = self.add_attr_colors(metadf)
            
        return metadf
    
    
    def add_attr_colors(self, metadf):
        cm = ColorMap(metadf)
        for col in metadf.columns:
            cm.add_color_column(col)
        return cm.metadf
    

    def add_cluster_color_and_shape(self, metadf):
        metadf = self.add_shape_column(metadf)
        metadf = self.add_color_column(metadf, 'cluster')

        # Create a new col for categorical attributes that matches a number to every cluster-attribute combination
        # Then map the new cols to colors
        for cattr in self.cat_attrs:
            colname = f'{cattr}_cluster'
            metadf[colname] = metadf.groupby([cattr, 'cluster']).ngroup()
            metadf = self.add_color_column(metadf, colname)

        return metadf


    def add_color_column(self, metadf, colname, use_skewed_colormap=True):
        # Add a color column to a df
        cm = ColorMap(metadf)
        metadf = cm.add_color_column(colname, use_skewed_colormap)
        metadf = cm.metadf
        return metadf
    
    
    def add_shape_column(self, metadf):
        # Add a shape column for the cluster column
        sm = ShapeMap(metadf)
        metadf = sm.add_shape_column()

        # assert len(sm.metadf) == self.nr_texts ################################
        return sm.metadf


class ShapeMap():
    '''
    Match clusters to shapes.
    Only implemented for 'cluster' column, use colors for other attributes.
    '''
    # Available shapes in plt, which work for nx
    SHAPES = ['o', 's', 'P', '*', 'd', 'p', '<', 'H', '>', 'v', 'h', '8', 'D', '^'] # 'X', 'x'
    
    def __init__(self, metadf):
        self.metadf = metadf
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def add_shape_column(self):
        '''
        Map each cluster to a shape. 
        Cluster with label 0 is mapped to the first list element, label 1 to the second element, and so on.
        If end of the list is reached, start from the beginning of the list.
        '''
        # Map clusters to indices of the SHAPES list
        self.metadf[f'clst_shape'] = self.metadf['cluster'] % len(self.SHAPES)
        # Map indices to shapes
        self.metadf[f'clst_shape'] = self.metadf[f'clst_shape'].apply(lambda x: self.SHAPES[x])
        # if self.metadf['cluster'].nunique() > len(self.SHAPES):
        #     self.logger.debug(f'Number of unique elements in "cluster" exceeds the number of shapes in SHAPES list. Different clusters have the same shape.')



class Colors():
    CMAP = plt.cm.get_cmap('seismic', lut=10000)
    CMAP_ALT = plt.cm.get_cmap('turbo', lut=10000)

    def __init__(self):
        pass


    @staticmethod
    def get_colors_discrete():
        '''
        Get discrete colors that are maximally (greedy) different from previous ones 
        The 'glasbey_hv' colormap contains 256 unique colors.
        '''
        colors = iter(cc.glasbey_bw_minc_20) # color palette with no greys
        return colors
    

    def get_colors_sequential(self, val, skewed=False):
        '''
        val: Number between 0 and 1
        '''
        if pd.isna(val):
            return 'green'
        if not skewed:
            color = self.CMAP(val)
        else:
            color = self.CMAP_ALT(val)
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
        self.cat_attrs = ['gender', 'author', 'canon-ascat', 'year-ascat']
        self.cat_attrs_combined = [f'{cattr}_cluster' for cattr in self.cat_attrs]


    def add_color_column(self, colname, use_skewed_colormap=True):
        if colname == 'gender':
            self.map_gender(colname)
        elif colname in self.cat_attrs + self.cat_attrs_combined + ['cluster']:
            self.map_categorical(colname)
        else:
            self.map_continuous(colname, use_skewed_colormap)

        # Transform format so it is compatible with pgv
        if self.pgv:
            self.metadf[f'{colname}_color'] = self.metadf[f'{colname}_color'].apply(self.color_for_pygraphviz)


    def map_gender(self, colname):
        gender_color_map = {0: 'blue', 1: 'red', 2: 'lightgreen', 3: 'yellow'}
        self.metadf[f'{colname}_color'] = self.metadf['gender'].map(gender_color_map)


    def map_categorical(self, colname):
        # Count the occurrences of each unique value in the specified column
        value_counts = self.metadf[colname].value_counts()
    
        # Create an iterator that cycles through colors
        colors = self.get_colors_discrete()

        # Create a mapping with cycling through colors for all unique elements
        # If an element occurs only once, set it to dark grey
        dark_grey =  (0.7, 0.7, 0.7) # RGB for dark gray
        color_mapping = {
            element: next(colors) if count > 1 else dark_grey
            for element, count in value_counts.items()
        }

        # Create a new column with the original column name appended with '_color'
        self.metadf[f'{colname}_color'] = self.metadf[colname].map(color_mapping)


    def map_continuous(self, colname, use_skewed_colormap=True):
        '''
        use_skewed_colormap: if True, alternative colormap is used if distribution of continuous values is skewed so that close values are better visually distinguishable
        '''
        # Scale values so that lowest value is 0 and highest value is 1
        scaled_col = pd.Series(minmax_scale(self.metadf[colname]))

        skewed = False
        skewness = skew(scaled_col)
        if abs(skewness) >= 3:
            skewed = True
        
        # Set skewed to False if skewed, but alternative color map should not be used
        if not use_skewed_colormap:
            skewed = False

        color_col = scaled_col.apply(lambda x: self.get_colors_sequential(x, skewed=skewed))
        self.metadf = self.metadf.assign(newcol=color_col.values).rename(columns={'newcol': f'{colname}_color'})



class CombinationInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.omit_default = ['metadf', 'param_comb', 'spars_param', 'omit_default', 'cluster_alg', 'spmx_path', 'clusterdf', 'order']
        self.clusterparams_to_string()       
        self.spars_to_string()


    def spars_to_string(self):
        add_zero_to_str = False
        if hasattr(self, 'sparsmode') and hasattr(self, 'spars_param'):
            if self.spars_param is None:
                self.sparsmode = self.sparsmode
            elif isinstance(self.spars_param, tuple):
                self.sparsmode = f'{self.sparsmode}-{self.spars_param[0]}-{self.spars_param[1]}'
            else:
                # Add trailing 0 to 0.9 (-> 0.90) to avoid naming conflics with 0.95
                if self.sparsmode == 'threshold' and self.spars_param == 0.9:
                    add_zero_to_str = True
                self.sparsmode = f'{self.sparsmode}-{str(self.replace_dot(self.spars_param))}'
                if add_zero_to_str:
                    self.sparsmode = f'{self.sparsmode}0'


    def clusterparams_to_string(self):
        if hasattr(self, 'cluster_alg'):
            if hasattr(self, 'param_comb'):
                self.clst_alg_params = None
                if bool(self.param_comb):
                    paramstr = '-'.join([f'{key}-{self.replace_dot(value)}' for key, value in self.param_comb.items()])
                    self.clst_alg_params = f'{self.cluster_alg}-{paramstr}'
                else:
                    self.clst_alg_params = f'{self.cluster_alg}'
                
                
    def replace_dot(self, value):
        # Use '%' to mark dot in float
        if isinstance(value, float):
            value = round(value, 3)
            return str(value).replace('.', '%')
        return value


    def as_string(self, sep='_', omit: List[str] = []):
        omit_lst = self.omit_default + omit

        filtered_values = []
        for key, value in self.__dict__.items():
            if key not in omit_lst:
                if value is not None and (not isinstance(value, dict)):
                    filtered_values.append(str(self.replace_dot(value)))
        
        return sep.join(filtered_values)
    

    def as_df(self, omit: List[str] = []):
        data = self.as_dict(omit)
        return pd.DataFrame([data], index=[0])


    def as_dict(self, omit: List[str] = []):
        omit_lst = self.omit_default + omit
        return {key: value for key, value in self.__dict__.items() if key not in omit_lst}    

    def add(self, key, value):
        # Add a new element to self.__dict__.
        self.__dict__[key] = value


    def drop(self, key):
        if key in self.__dict__:
            del self.__dict__[key]
            logging.info(f"Key '{key}' removed successfully.")
        else:
            logging.info(f"Key '{key}' not found.")
