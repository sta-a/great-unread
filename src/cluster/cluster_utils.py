# %%

import matplotlib.pyplot as plt
import pandas as pd
import os
import colorcet as cc
import itertools
import sys
sys.path.append("..")
from sklearn.preprocessing import minmax_scale
from copy import deepcopy
from utils import DataHandler, DataLoader, TextsByAuthor

import logging
logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
ticker_logger = logging.getLogger('matplotlib.ticker')
ticker_logger.setLevel(logging.WARNING)

class MetadataHandler(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='similarity', data_type='png')


    def get_metadata(self):
        canon = DataLoader(self.language).prepare_metadata(type='canon')
        gender = DataLoader(self.language).prepare_metadata(type='gender')
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
        df_list = [canon, gender, features, author, year]
        merged_df = pd.DataFrame()
        for df in df_list:
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='inner', validate='1:1')

        assert len(merged_df) == self.nr_texts
        return merged_df

    # def get_metadata(self):
    #     if self.attr == 'canon':
    #         df = DataLoader(self.language).prepare_metadata(type=self.attr)
    #     # discretize
    #     # file_group_mapping['scores'].apply(lambda x: 'r' if x > threshold else 'b')

    #     elif self.attr == 'gender':
    #         df = DataLoader(self.language).prepare_metadata(type=self.attr)

    #     elif self.attr == 'features':
    #         df = DataLoader(self.language).prepare_features() # file name is index

    #     elif self.attr == 'author':
    #         author_filename_mapping = TextsByAuthor(self.language).author_filename_mapping
    #         df = pd.DataFrame([(author, work) for author, works in author_filename_mapping.items() for work in works],
    #                         columns=['author', 'file_name'])
    #         df = df.set_index('file_name', inplace=False)

    #     elif self.attr == 'year':
    #         fn = [os.path.splitext(f)[0] for f in os.listdir(self.text_raw_dir) if f.endswith('.txt')]
    #         df = pd.DataFrame({'file_name': fn})
    #         df['year'] = df['file_name'].str[-4:].astype(int)
    #         df = df.set_index('file_name', inplace=False)
        
    #     return df
    
    
    def add_color(self, df, attr, col_name=None):
        # Add color column
        self.cm = ColorMap(df, attr, col_name)
        df = self.cm.get_color_map()

        assert len(df) == self.nr_texts
        return df
    

class Colors():
    def __init__(self):
        # Lut: the number of colors that are generated from the color map
        # Set to a high value to get a new color for values that are close together
        self.cmap = plt.cm.get_cmap('BuPu', lut=10000) # gist_yarg


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
        color = self.cmap(val)
        return color
    
    def color_for_pygraphviz(self, color):
        # HSV needs to be passed as a string for pygraphviz
        return ' '.join(str(i) for i in color)


class ColorMap(Colors):
    '''
    Map the column with name 'attr' of metadf to a color.
    'attr' is either a metadata attribute (gender, author etc.), or the cluster assingments.
    '''
    def __init__(self, metadf, attr, col_name):
        super().__init__()
        self.metadf = metadf
        self.attr = attr
        self.col_name = col_name
        if self.col_name is None:
            self.col_name = self.attr
        self.pgv = False


    def get_color_map(self):
        if self.attr == 'gender':
            self.map_gender()
        elif (self.attr == 'cluster') or (self.attr == 'author'):
            self.map_categorical()
        elif (self.attr == 'canon') or (self.attr == 'year') or (self.attr == 'features'):
            self.map_continuous()

        # Color column contains unhashable lists
        # Convert to tuples
        color_col = self.metadf['color'].apply(tuple)
        assert self.metadf[self.col_name].nunique() == color_col.nunique()

        # Transform format so it is compatible with pgv
        if self.pgv:
            self.metadf['color'] = self.metadf['color'].apply(self.color_for_pygraphviz)

        return self.metadf


    def map_gender(self):
        gender_color_map = {'f': 'red', 'm': 'blue', 'b': 'yellow', 'a': 'lightgreen'}
        self.metadf['color'] = self.metadf['gender'].map(gender_color_map)
    
    def map_categorical(self):
        self.metadf['color'] = self.metadf[self.col_name].map(dict(zip(sorted(self.metadf[self.col_name].unique()), self.get_colors_discrete())))

    def map_continuous(self):
        # Scale values so that lowest value is 0 and highest value is 1
        scaled_col = minmax_scale(self.metadf[self.col_name])
        self.metadf['color'] = pd.Series(scaled_col).apply(self.get_colors_sequential)
    

class ShapeMap():
    # Available shapes in nx
    SHAPES = ['egg', 'invtrapezium', 'star', 'cylinder', 'cds', 'terminator', 'box', 'tripleoctagon','Mdiamond', 'Mcircle', 'Mdiamond', 'Mcircle', 'Msquare', 'circle', 'diamond', 'doublecircle', 'doubleoctagon', 'ellipse', 'hexagon', 'house', 'invhouse', 'invtriangle', 'none', 'octagon', 'oval', 'parallelogram', 'pentagon', 'plaintext', 'point', 'polygon', 'rectangle', 'septagon', 'square', 'tab', 'trapezium', 'triangle']
    
    def __init__(self, metadf):
        self.metadf = metadf

    def map_shape(self):
        # Map each cluster to a shape. 
        # Cluster with label 0 is mapped to the first list element, label 1 to the second element, and so on.
        # If end of the list is reached, start from the beginning of the list.

        # Map clusters to shapes
        self.metadf['shape'] = self.metadf['cluster'] % len(self.SHAPES)
        self.metadf['shape'] = self.metadf['shape'].apply(lambda x: self.SHAPES[x])
        return self.metadf



class CombinationInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if 'param_comb' in kwargs:
            self.param_comb = self.paramcomb_to_string(self.param_comb)

    def paramcomb_to_string(self, param_comb):
        return '-'.join([f'{key}-{value}' for key, value in param_comb.items()])

    def as_string(self, omit=None):
        if omit is None:
            omit = []
        return '_'.join(str(value) for key, value in self.__dict__.items() if key not in omit)

    def as_df(self):
        # data = {key: [value] for key, value in self.__dict__.items()}
        # return pd.DataFrame([data], index=[0])
        return pd.DataFrame([vars(self)], index=[0])

    def as_dict(self):
        return self.__dict__
    


# %%
# class MxViz(DataHandler):


#     def draw_mds(self, clusters):
#         print(f'Drawing MDS.')
#         df = MDS(n_components=2, dissimilarity='precomputed', random_state=6, metric=True).fit_transform(self.mx)
#         df = pd.DataFrame(df, columns=['comp1', 'comp2'], index=self.mx.index)
#         df = df.merge(self.file_group_mapping, how='inner', left_index=True, right_on='file_name', validate='one_to_one')
#         df = df.merge(clusters, how='inner', left_on='file_name', right_index=True, validate='1:1')

#         def _group_cluster_color(row):
#             color = None
#             if row['group_color'] == 'b' and row['cluster'] == 0:
#                 color = 'darkblue'
#             elif row['group_color'] == 'b' and row['cluster'] == 1:
#                 color = 'royalblue'
#             elif row['group_color'] == 'r' and row['cluster'] == 0:
#                 color = 'crimson'
#             #elif row['group_color'] == 'r' and row['cluster'] == 0:
#             else:
#                 color = 'deeppink'
#             return color

#         df['group_cluster_color'] = df.apply(_group_cluster_color, axis=1)


#         fig = plt.figure(figsize=(5,5))
#         ax = fig.add_subplot(1,1,1)
#         plt.scatter(df['comp1'], df['comp2'], color=df['group_cluster_color'], s=2, label="MDS")
#         plt.title = self.plot_name
#         self.save(plt, 'kmedoids-MDS', dpi=500)



# %%
