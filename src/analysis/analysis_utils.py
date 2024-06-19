
# %%

import pandas as pd
import os
import re
import networkx as nx
import pickle
import pandas as pd

from tqdm import tqdm

import sys
sys.path.append("..")
from utils import DataHandler
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


        cluster_path_string = '/cluster/scratch/stahla/data'
        if cluster_path_string in info.spmx_path:
            info.spmx_path = info.spmx_path.replace(cluster_path_string, '/home/annina/scripts/great_unread_nlp/data')


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
            print(file_name)

            # Create the new filename by replacing 0%9 with 0%90
            pattern = re.compile(r'0%9$')
            if pattern.search(file_name):
                file_name = pattern.sub('0%90', file_name)
                print(file_name)
                print('\n\n-------------------------')


            nx.write_weighted_edgelist(graph, os.path.join(outdir, f'{file_name}.csv'), delimiter=sep)
        
        if index_mapping is not None:
            index_mapping.to_csv(os.path.join(outdir, f'index-mapping.csv'), header=True, index=False)


params = [('sparsification_edgelists', False, True, ','), ('sparsification_edgelists_s2v', True, True, ' '), ('sparsification_edgelists_labels', False, False, ',')]

# for p in params:
#     pklmxs_to_edgelist(p)


 # %%
