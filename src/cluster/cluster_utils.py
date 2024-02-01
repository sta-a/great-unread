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
import matplotlib.lines as mlines

import matplotlib.patches as mpatches
from matplotlib import markers
import textwrap
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
        self.cat_attrs = ['gender', 'author']


    def get_metadata(self, add_color=False):
        gender = DataLoader(self.language).prepare_metadata(type='gender')
        gender = gender[['gender']]
        gender['gender'] = gender['gender'].map({'m': 0, 'f': 1, 'a': 2, 'b': 3})

        canon = DataLoader(self.language).prepare_metadata(type='canon')

        # Load features scaled to range between 0 and 1
        features = DataLoader(self.language).prepare_features(scale=True)
        # Check if all feature values are between 0 and 1, ignore nan
        features_test = features.fillna(0)
        assert ((features_test >= 0) & (features_test <= 1) | (features_test.isin([0, 1]))).all().all()


        out_of_range_indices = features[(features < 0)].stack().index

        # Print the elements with row and column indices
        for row_idx, col_idx in out_of_range_indices:
            print(f"Element at ({row_idx}, {col_idx}): {features.loc[row_idx, col_idx]}")



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
        
        # Remove '_full' from column names
        metadf = metadf.rename(columns=lambda x: x.rstrip('_full'))
        # Replace '_' in feature names with '-'
        metadf = metadf.rename(columns=lambda x: x.replace('_', '-'))

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


    def add_color_column(self, metadf, colname):
        # Add a color column to a df
        cm = ColorMap(metadf)
        metadf = cm.add_color_column(colname)
        metadf = cm.metadf

        assert len(metadf) == self.nr_texts
        return metadf
    
    
    def add_shape_column(self, metadf):
        # Add a shape column for the cluster column
        sm = ShapeMap(metadf)
        metadf = sm.add_shape_column()

        assert len(sm.metadf) == self.nr_texts
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

    def get_colors_sequential(self, val):
        '''
        val: Number between 0 and 1
        '''
        if pd.isna(val):
            return 'green'
        color = self.CMAP(val)
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
        self.cat_attrs = ['gender', 'author']
        self.cat_attrs_combined = [f'{cattr}_cluster' for cattr in self.cat_attrs]


    def add_color_column(self, colname):
        if colname == 'gender':
            self.map_gender(colname)
        elif colname in self.cat_attrs + self.cat_attrs_combined + ['cluster']:
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
        # If an element occurs only once, set it to dark grey
        darkgrey_rgb = to_rgba('darkgrey')
        color_mapping = {
            element: next(colors) if count > 1 else darkgrey_rgb
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
        self.omit_default = ['metadf', 'param_comb', 'spars_param', 'omit_default', 'cluster_alg', 'spmx_path', 'clusterdf', 'order']
        self.clusterparams_to_string()       
        self.spars_to_string()


    def spars_to_string(self):
        if hasattr(self, 'sparsmode') and hasattr(self, 'spars_param'):
            if self.spars_param is None:
                self.sparsmode = self.sparsmode
            elif isinstance(self.spars_param, tuple):
                self.sparsmode = f'{self.sparsmode}-{self.spars_param[0]}-{self.spars_param[1]}'
            else:
                self.sparsmode = f'{self.sparsmode}-{str(self.replace_dot(self.spars_param))}'


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
            # Print a log message if the key is not in the dictionary
            logging.info(f"Key '{key}' not found.")


        
class VizBase(DataHandler):
    def __init__(self, language, cmode, info, plttitle):
        super().__init__(language, output_dir='similarity', data_type='png')
        self.cmode = cmode
        self.info = info
        self.plttitle = plttitle
        self.fontsize = 12
        self.add_subdir(f'{self.cmode}top')
        self.cat_attrs = ['gender', 'author']
        self.is_cat = False
        if self.info.attr in self.cat_attrs:
            self.is_cat = True


    def add_suptitle(self, width=100, **kwargs):  
        if self.plttitle is not None:
            plt.suptitle(textwrap.fill(self.plttitle, width=width), fontsize=self.fontsize, **kwargs)


    def save_plot(self, plt, file_name=None, file_path=None):
        self.save_data(data=plt, data_type=self.data_type, subdir=True, file_name=file_name, file_path=file_path)
   

    def get_path(self, name='viz', omit: List[str] = [], data_type='pkl'):
        file_name = f'{name}-{self.info.as_string(omit=omit)}.{data_type}'
        return self.get_file_path(file_name, subdir=True)
    

    def add_cbar(self, ax):
        # Create a color bar from a color map
        # The color map is not used in any matplotlib functions (like for a heatmap), therefore the bar has to be created manually.
        # Create a ScalarMappable with the color map
        cmap = Colors.CMAP
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Create a color bar using the ScalarMappable
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.ax.tick_params(axis='both', which='major', labelsize=self.fontsize)

        # Set a label for the color bar
        # cbar.set_label('Color Bar Label', rotation=270, labelpad=15)

        # Remove ticks from the color bar
        # cbar.set_ticks([])

        # Adjust the layout to make room for the color bar
        # fig_cbar.tight_layout()        


    def add_legend(self, fig_or_ax, attr, label='size', use_shapes=False, loc='upper right', boxx=1.05, boxy=1, boxwidth=0.2, boxheight=0.4, fontsize=None, markersize=8):
        bbox_to_anchor = (boxx, boxy, boxwidth, boxheight)
        if fontsize is None:
            fontsize = self.fontsize

        mapping = {}
        for unique_attr in self.df[attr].unique().tolist():
            cdf = self.df[self.df[attr] == unique_attr]
            attribute = cdf.iloc[0][f'{attr}_color']
            if use_shapes:
                shape = cdf.iloc[0]['clst_shape']
            else:
                shape = 'o'

            if len(cdf) > 1:
                mapping[unique_attr] = (attribute, shape, len(cdf))

        # Keep the 10 most frequent elements
        mapping = dict(sorted(mapping.items(), key=lambda item: item[1][2], reverse=True))
        mapping = {k: v for k, v in list(mapping.items())[:10]}


        # Create legend patches
        legend_patches = []
        for unique_attr, (attribute, shape, count) in mapping.items():
            if label == 'size':
                clabel = f'{count}'
            elif label == 'attr':
                # Underscores cannot be used in labels because they increase the space between the labels in the legend
                # This causes the two legends to not be aligned
                
                # If attr is author name, get only the first letter of the first name to keep legend short
                if '_' in unique_attr:
                    name_parts = unique_attr.split("_")
                    clabel = f'{name_parts[0]}{name_parts[1][0]} ({attribute})' ####################
                else:
                    clabel = f'{unique_attr} ({attribute})'
                
            legend_patches.append(mlines.Line2D([], [], color=attribute, marker=shape, label=clabel, linestyle='None', markersize=markersize))

        fig_or_ax.legend(handles=legend_patches, labelspacing=0.5, loc=loc, bbox_to_anchor=bbox_to_anchor, fontsize=fontsize)



    def add_subplot_titles(self, attrax, clstax, shapeax, combax):
        attrax.set_title('Attribute', fontsize=self.fontsize)
        clstax.set_title('Cluster', fontsize=self.fontsize)
        shapeax.set_title('Attributes and clusters (shapes)', fontsize=self.fontsize)
        if combax is not None:
            combax.set_title('Attributes and clusters (combined)', fontsize=self.fontsize)
