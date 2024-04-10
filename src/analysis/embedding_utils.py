

import sys
sys.path.append("..")
import pandas as pd
import os
import networkx as nx
from utils import DataHandler
from tqdm import tqdm
from itertools import product


class EdgelistHandler(DataHandler):
    def __init__(self, language, output_dir, edgelist_dir='sparsification_edgelists'):
        super().__init__(language, output_dir=output_dir)
        self.edgelist_dir = os.path.join(self.data_dir, 'similarity', self.language, edgelist_dir)
        self.edgelists = [file for file in os.listdir(self.edgelist_dir) if 'index-mapping' not in file and file.endswith('.csv')]
        spars_examples = [
             'cosinedelta-1000_threshold-0%8', 
             'both_threshold-0%9.', # dot to avoid confusion with 0%95
             'cosinesim-1000_threshold-0%95', 
             'full_authormin', 
             'cosinesim-5000_authormax', 
             'sqeuclidean-500_simmel-3-10', 
             'both_simmel-4-6', 
             'burrows-500_simmel-5-10',
             'eder-5000_simmel-7-10', 
             ]
        self.edgelists = [
            filename for filename in self.edgelists
            if any(substring in filename for substring in spars_examples)
        ]
  

        self.nr_mxs = 58
        self.nr_spars = 9
        # assert len(self.edgelists) == self.nr_mxs * self.nr_spars
        self.remove_iso = True


    def print_graph_info(self, graph):
        print(f'--------------\n'
                f'nx Graph Overview: \n'
                f'Nr. of nodes: {graph.number_of_nodes()} \n'
                f'Nr. of edges: {graph.number_of_edges()} \n'
                f'is weighted: {nx.is_weighted(graph, weight="weight")} \n'
                f'is directed: {nx.is_directed(graph)} \n'
                f'Nr. of Selfloops: {nx.number_of_selfloops(graph)} \n'
                f'Nr. isolated nodes: {len(list(nx.isolates(graph)))} \n'
                f'--------------\n')


    def network_from_edgelist(self, file_path, print_info=False):
        if self.remove_iso:
                self.logger.debug(f'Isolated nodes are removed from the graph.')

        if 'threshold' in file_path:
                graph = nx.read_weighted_edgelist(file_path, delimiter=',')
        else:
                graph = nx.read_weighted_edgelist(file_path, delimiter=',', create_using=nx.DiGraph()) ###### set node type to int???

        # An isolated node has degree zero. A single node with only a self-loop is not an isolated node.
        graph.remove_edges_from(nx.selfloop_edges(graph))
        if self.remove_iso:
                nodes_iso = list(nx.isolates(graph))
                graph.remove_nodes_from(nodes_iso)

        if print_info:
                self.print_graph_info(graph)
        return graph




class NetworkStats(EdgelistHandler):
    def __init__(self, language, output_dir='analysis'):
        super().__init__(language, output_dir=output_dir)
        self.stats_path = os.path.join(self.output_dir, f'network-stats_remove-iso-{self.remove_iso}.csv')


    def get_network_stats(self):
        stats = []

        for fn in tqdm(self.edgelists):
        # if 'braycurtis-2000' in fn:
            if self.language == 'eng' and ('cosinesim-500_simmel-4-6' in fn or 'cosinesim-500_simmel-7-10' in fn): # no edges
                continue
            print(fn)
            network = self.network_from_edgelist(os.path.join(self.edgelist_dir, fn))
            if nx.is_directed(network): # nx.connected_components only implemented for undirected
                network = network.to_undirected()
            components = list(nx.connected_components(network))
            component_sizes = sorted([len(component) for component in components])
            all_sizes = ','.join([str(x) for x in component_sizes])

            largest_component = max(components, key=len)
            largest_component_graph = network.subgraph(largest_component)
            aspl = nx.average_shortest_path_length(largest_component_graph)
            diam = nx.diameter(largest_component_graph)
            
            nrs = [fn, min(component_sizes), max(component_sizes), len(component_sizes), all_sizes, aspl, diam]
            stats.append(nrs)

        stats = pd.DataFrame(stats, columns=['file_name', 'min_component_size', 'max_component_size', 'nr_components', 'all_size', 'av_shortest_path_length', 'diameter'])
        stats.to_csv(self.stats_path, header=True, index=False)


    def get_network_selfloops(self):
        stats = []
        for fn in self.edgelists:
                if self.language == 'eng' and ('cosinesim-500_simmel-4-6' in fn or 'cosinesim-500_simmel-7-10' in fn): # no edges
                    continue
                print(fn)
                network = self.network_from_edgelist(os.path.join(self.edgelist_dir, fn))
                if nx.is_directed(network): # nx.connected_components only implemented for undirected
                    network = network.to_undirected()
                components = list(nx.connected_components(network))
                component_sizes = sorted([len(component) for component in components])

                nr_selfloops = nx.number_of_selfloops(network)

                all_sizes = ','.join([str(x) for x in component_sizes])
                nrs = [fn, nr_selfloops, len(component_sizes), all_sizes]
                stats.append(nrs)

        stats = pd.DataFrame(stats, columns=['file_name', 'has_selfloops', 'nr_components', 'all_size'])
        loop_path = os.path.join(self.output_dir, 'network_selfloops.csv')
        stats.to_csv(loop_path, header=True, index=False)


    def analyze_network_stats(self):
        df = pd.read_csv(self.stats_path, header=0)

        df['mxnames'] = df['file_name'].apply(lambda x: x.split('_')[0])
        df['sparsnames'] = df['file_name'].apply(lambda x: x.split('_')[1].split('.')[0])

        # Get unique values of 'x' and 'y'
        mxnames = df['mxnames'].unique().tolist()
        sparsnames = df['sparsnames'].unique().tolist()
        assert len(mxnames) == self.nr_mxs
        assert len(sparsnames) == self.nr_spars


        grouped = df.groupby('sparsnames').agg({
        'min_component_size': ['max', 'min'],
        'max_component_size': ['max', 'min'],
        'nr_components': ['max', 'min'],
        'av_shortest_path_length': ['max', 'min'],
        'diameter': ['max', 'min']
        })

        # Rename columns for clarity
        grouped.columns = ['max_min_component_size', 'min_min_component_size',
                            'max_max_component_size', 'min_max_component_size',
                            'max_nr_components', 'min_nr_components',
                            'max_av_shortest_path_length', 'min_av_shortest_path_length',
                            'max_diameter', 'min_diameter']

        grouped.to_csv(os.path.join(self.output_dir, 'network_stats_analysis.csv'), header=True, index=True)



class EmbeddingBase(EdgelistHandler):
    '''
    Enable n2v/s2v conda environment!
    This class is a base for creating s2v and n2v embeddings.

    noedges eng: cosinesim-500_simmel-4-6
    cosinesim-500_simmel-7-10
    '''
    def __init__(self, language, output_dir, edgelist_dir):
        super().__init__(language, output_dir=output_dir, edgelist_dir=edgelist_dir)
        self.add_subdir('embeddings')


    def load_embeddings(self, file_name):
        index_mapping = pd.read_csv(os.path.join(self.edgelist_dir, 'index-mapping.csv'), header=0)

        inpath = os.path.join(self.subdir, f'{file_name}.embeddings')
        df = pd.read_csv(inpath, skiprows=1, sep=' ')
        
        # Use the first column with the ID as index
        df = df.set_index(df.columns[0])

        # Rename columns as col1, col2, ...
        df.columns = [f'col{i}' for i in range(1, len(df.columns) + 1)]

        # Map int IDs to file names
        df = df.merge(index_mapping, left_index=True, right_on='new_index', validate='1:1')
        df = df.set_index('original_index')
        df = df.drop('new_index', axis=1)
        df.index = df.index.rename('file_name')
        return df
    

    def get_param_string(self, kwargs):
        d = {key.replace('-', ''): value for key, value in kwargs.items()} # remove '-', such as in 'walk-length'
        return '_'.join(f'{key}-{value}' for key, value in d.items())
    

    def get_embedding_path(self, fn, kwargs):
        if '.csv' in fn:
            fn = os.path.splitext(fn)[0]
        param_string = self.get_param_string(kwargs)
        return os.path.join(self.subdir, f'{fn}_{param_string}.embeddings')
    

    def create_data(self, kwargs={}):
        for fn in self.edgelists:
                print(fn)
                if self.language == 'eng' and ('cosinesim-500_simmel-4-6' in fn or 'cosinesim-500_simmel-7-10' in fn): # no edges
                    continue

                embedding_path = self.get_embedding_path(fn, kwargs)
                if not os.path.exists(embedding_path):
                    self.create_embeddings(fn, kwargs)


    def get_param_combinations(self):
        params = self.get_params()

        # Check if params is empty dict
        if not bool(params):
                return [{}]
        
        else:
            # Create a list of dicts with format param_name: param_value.
            param_combs = []
            combinations_product = product(*params.values())
            # Iterate through all combinations and pass them to the function
            for combination in combinations_product:
                param_combs.append(dict(zip(params.keys(), combination)))

            return param_combs
        
        
    def run_combinations(self):
        param_combs = self.get_param_combinations()
        for comb in param_combs:
            self.create_data(comb)


# for language in ['eng', 'ger']:
#     ns = NetworkStats(language)
#     ns.get_network_stats()
# %%
