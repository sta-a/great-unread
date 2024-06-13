
# %%

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from copy import deepcopy
from typing import List
import networkx as nx
import pickle
import pandas as pd

import matplotlib.lines as mlines
import textwrap
from tqdm import tqdm

import sys
sys.path.append("..")
from utils import DataHandler
from cluster.cluster_utils import Colors
from cluster.combinations import InfoHandler
from cluster.network import NXNetwork
import logging
logging.basicConfig(level=logging.DEBUG)



class NoedgesLoader(DataHandler):
    def __init__(self, language, output_dir='similarity'):
        super().__init__(language, output_dir)
        noedges_path = os.path.join(self.output_dir, 'nk_noedges.txt')

        
    def get_noedges_list(self):
        if not os.path.exists(self.noedges_path):
            unique_lines = []
    
        else:
            lines = []
            with open(self.noedges_path, 'r') as file:
                lines = file.readlines()
            unique_lines = list(set(lines))
            unique_lines = [line.strip() for line in unique_lines if line.strip()]

            print('noedges')
            for i in unique_lines:
                print(i)

        return unique_lines
    

class VizBase(DataHandler):
    def __init__(self, language, output_dir='analysis', cmode='nk', info=None, plttitle=None, exp=None, by_author=False):
        super().__init__(language, output_dir=output_dir, data_type='png')
        self.cmode = cmode
        self.info = info
        self.plttitle = plttitle
        self.exp = exp
        self.by_author = by_author
        self.fontsize = 12
        self.add_custom_subdir()

        self.special_cols = ['cluster', 'clst_shape', 'gender_cluster', 'author_cluster', 'x', 'y', 'pos']
        self.key_attrs = ['author', 'gender', 'year', 'canon']
        self.cat_attrs = ['gender', 'author']

        self.by_author_attrs = ['canon-max', 'canon-min']
        if self.by_author:
            self.key_attrs.remove('author')
            self.key_attrs.extend(self.by_author_attrs)

        self.is_cat = False
        if (self.info is not None) and (self.info.attr in self.cat_attrs):
            self.is_cat = True

        self.needs_cbar = self.check_cbar()


    def add_custom_subdir(self):
        self.add_subdir(f"{self.cmode}_{self.exp['name']}")


    def get_metadf(self):
        self.df = deepcopy(self.info.metadf)


    def check_cbar(self):
        # Check if any continuous attributes are shown.
        # If yes, a cbar is necessary.
        cbar = False
        if not self.is_cat:
            cbar = True
        return cbar


    def add_text(self, ax, x=0, y=0, width=30):  
        ax.text(x=x, y=y, s=textwrap.fill(self.plttitle, width), fontsize=self.fontsize)


    def save_plot(self, plt):
        self.save_data(data=plt, data_type=self.data_type, file_name=None, file_path=self.vizpath, plt_kwargs={'dpi': 300})


    def get_path(self, name='viz', omit: List[str]=[], data_type=None):
        if data_type is None:
            data_type = self.data_type
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
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, location='left')
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
        self.df[attr] = self.df[attr].astype(str) # Enforce that gender column is treated as string
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
                    clabel = f'{name_parts[0]}{name_parts[1][0]} ({count})'
                else:
                    clabel = f'{unique_attr} ({count})'
                
            legend_patches.append(mlines.Line2D([], [], color=attribute, marker=shape, label=clabel, linestyle='None', markersize=markersize))

        fig_or_ax.legend(handles=legend_patches, labelspacing=0.5, loc=loc, bbox_to_anchor=bbox_to_anchor, fontsize=fontsize)


    def get_ax(self, ix):
        return self.axs[ix[0], ix[1]]


    def add_subtitles(self, attrix, clstix, shapeix, combix=None):
        self.get_ax(attrix).set_title(f'Attribute: {self.info.attr}', fontsize=self.fontsize)
        self.get_ax(clstix).set_title('Clusters', fontsize=self.fontsize)
        self.get_ax(shapeix).set_title('Attribute (color) and clusters (shapes)', fontsize=self.fontsize)
        if combix is not None:
            self.get_ax(combix).set_title('Attribute and clusters (combined)', fontsize=self.fontsize)


def main_attributes_crosstable():
    eng = '/home/annina/scripts/great_unread_nlp/data/similarity/eng/metadf.csv'
    ger = '/home/annina/scripts/great_unread_nlp/data/similarity/ger/metadf.csv'
    df = pd.read_csv(ger, header=0)
    correlation = df['year'].corr(df['canon'])


    # Create cross tables for all combinations
    cross_tables = {}
    categorical_cols = ['gender', 'author']
    continuous_cols = ['year', 'canon']


    df['century'] = pd.cut(df['year'], bins=range(1600, 2001, 100), labels=['17th', '18th', '19th', '20th'])
    df[['century', 'year']]
    df['canon_chunk'] = pd.cut(df['canon'], bins=5, labels=['lowest', 'low', 'medium', 'high', 'highest'])
    cross_tables = {}
    categorical_cols = ['gender','century', 'canon_chunk'] # , 'author', 

    for cat_col1 in categorical_cols:
        for cat_col2 in categorical_cols:
            if cat_col1 != cat_col2:  # Exclude identical combinations
                cross_tables[(cat_col1, cat_col2)] = pd.crosstab(df[cat_col1], df[cat_col2])

    # Display cross tables
    for key, table in cross_tables.items():
        print(f"Cross Table for {key}:")
        print(table)



def map_indices_to_numbers(similarity_matrix, map_index_to_int=True):
    """
    Map string indices to numbers in a Pandas DataFrame.
    
    Parameters:
        similarity_matrix (pd.DataFrame): DataFrame with string indices.
        
    Returns:
        pd.DataFrame: DataFrame with string indices replaced by numbers.
        dict: Mapping of string indices to numbers.
    """
    # Check if alphabetically sorted
    assert list(similarity_matrix.index) == sorted(similarity_matrix.index)
    assert list(similarity_matrix.columns) == sorted(similarity_matrix.index)
    assert list(similarity_matrix.index) == list(similarity_matrix.columns)

    if not map_index_to_int:
        return similarity_matrix, None


    # Create a mapping of string indices to numbers
    index_mapping = {index: i+1 for i, index in enumerate(similarity_matrix.index)}
    
    # Map string indices to numbers for both index and columns
    mapped_matrix = similarity_matrix.rename(index=index_mapping).rename(columns=index_mapping)

    index_mapping = pd.DataFrame(list(index_mapping.items()), columns=['original_index', 'new_index'])
    
    return mapped_matrix, index_mapping


def load_spmx_from_pkl(spmx_path):
    '''
    Load the sparsified matrices from pkl.
    '''
    with open(spmx_path, 'rb') as f:
        simmx = pickle.load(f)

    simmx = simmx.mx

    return simmx


def info_to_mx_and_edgelist():
    outdir = '/home/annina/scripts/great_unread_nlp/src/networks_to_embeddings'
    d = {'eng': 'sqeuclidean-2000_simmel-3-10_louvain-resolution-0%1', 'ger': 'full_simmel-3-10_louvain-resolution-0%01'} # info strings for interesting combinations
    for language, info in d.items():
        ih = InfoHandler(language=language, add_color=False, cmode='nk')

        info = ih.load_info(info)
        print(info.as_string())
        attributes = ih.metadf['canon']
        attributes = pd.DataFrame({'index': attributes.index, 'canon': attributes.values})

        spmx_path = info.spmx_path
        network = NXNetwork(language=language, path=spmx_path)
        mapped_matrix, index_mapping = map_indices_to_numbers(network.mx)
        print('Mapped DataFrame:')
        print(mapped_matrix)
        print('\nIndex Mapping:')
        print(index_mapping)

        attributes = attributes.merge(index_mapping, left_on='index', right_on='original_index', how='inner', validate='1:1')
        attributes = attributes[['new_index', 'canon']]
        attributes = attributes.rename(columns={'canon': 'score', 'new_index': 'index'})
        assert len(attributes) == len(mapped_matrix)

        if mapped_matrix.equals(mapped_matrix.T):
            graph = nx.from_pandas_adjacency(mapped_matrix)
        else:
            graph = nx.from_pandas_adjacency(mapped_matrix, create_using=nx.DiGraph) 
        # Remove selfloops
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # mapped_matrix.to_csv(os.path.join(outdir, f'weightmatrix-{language}-{info.as_string()}'), header=True, index=True)
        attributes.to_csv(os.path.join(outdir, f'attributes-{language}-{info.as_string()}.csv'), header=True, index=False)
        index_mapping.to_csv(os.path.join(outdir, f'index-mapping-{language}-{info.as_string()}.csv'), header=True, index=False)
        nx.write_weighted_edgelist(graph, os.path.join(outdir, f'edgelist_{language}-{info.as_string()}.csv'), delimiter=',')




def pklmxs_to_edgelist(params):
    '''
    Rewrite all sparsified matrices, which are in pkl format, as edge lists. Map string indices to numbers.
    exclude_iso_nodes: if True, isolated nodes are not in the edgelist.
    '''
    spars_dir, exclude_iso_nodes, map_index_to_int, sep = params
    print(spars_dir, exclude_iso_nodes, sep)

    for language in ['eng', 'ger']:
        # indir = f'/home/annina/scripts/great_unread_nlp/data/similarity/{language}/sparsification'
        # outdir = f'/home/annina/scripts/great_unread_nlp/data/similarity/{language}/{spars_dir}'
        indir = f'/home/annina/scripts/great_unread_nlp/data_author/similarity/{language}/sparsification'
        outdir = f'/home/annina/scripts/great_unread_nlp/data_author/similarity/{language}/{spars_dir}'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        mxpaths = [os.path.join(indir, file) for file in os.listdir(indir) if file.endswith('.pkl')]
        prev_idx_mapping = None
        for spmx_path in tqdm(mxpaths):
            network = NXNetwork(language=language, path=spmx_path)
            mapped_matrix, index_mapping = map_indices_to_numbers(network.mx.mx, map_index_to_int)

            if exclude_iso_nodes:
                diag_val = 0
            else:
                diag_val = 1
            assert mapped_matrix.index.equals(mapped_matrix.columns)
            for i in range(len(mapped_matrix)):
                for j in range(len(mapped_matrix)):
                    if i == j:
                        mapped_matrix.iloc[i,j] = diag_val


            if prev_idx_mapping is None:
                prev_idx_mapping = index_mapping
            else:
                assert index_mapping.equals(prev_idx_mapping)

            if mapped_matrix.equals(mapped_matrix.T):
                graph = nx.from_pandas_adjacency(mapped_matrix)
            else:
                graph = nx.from_pandas_adjacency(mapped_matrix, create_using=nx.DiGraph)

            
            # Remove selfloops so that isolated nodes are not included
            if exclude_iso_nodes:
                graph.remove_edges_from(nx.selfloop_edges(graph))


            file_name = os.path.splitext(os.path.basename(spmx_path))[0]
            file_name = file_name.replace('sparsmx-', '')

            nx.write_weighted_edgelist(graph, os.path.join(outdir, f'{file_name}.csv'), delimiter=sep)
        
        if index_mapping is not None:
            index_mapping.to_csv(os.path.join(outdir, f'index-mapping.csv'), header=True, index=False)


params = [('sparsification_edgelists_labels', False, False, ',')]
params = [('sparsification_edgelists', False, True, ','), ('sparsification_edgelists_s2v', True, True, ' ')]

for p in params:
    pklmxs_to_edgelist(p)


 # %%
