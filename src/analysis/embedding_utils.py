import sys
sys.path.append("..")
import pandas as pd
import os
import networkx as nx
from utils import DataHandler
from tqdm import tqdm
from itertools import product


class EdgelistHandler(DataHandler):
    def __init__(self, language, output_dir, edgelist_dir='sparsification_edgelists', mode=None, by_author=False):
        '''
        mode: 
            - if None: embeddings for all networks in edgelist dir
            - if 'run': embeddings for interesting networks
            - if 'params': embeddings for example networks with many parameter combinations
        '''
        super().__init__(language, output_dir=output_dir, load_doc_paths=False)
        self.mode = mode
        self.by_author = by_author
        assert self.mode in [None, 'run', 'params', 'bestparams']

        self.edgelist_dir = os.path.join(self.data_dir, 'similarity', self.language, edgelist_dir)
        self.edgelists = [file for file in os.listdir(self.edgelist_dir) if 'index-mapping' not in file and file.endswith('.csv')]

        if self.mode == 'params':
            if not self.by_author:
                if self.language == 'eng':

                    self.examples_full = {
                        'full_authormax': 'Bronte_Charlotte_Jane-Eyre_1847', 
                        'full_threshold-0%8': 'Bronte_Charlotte_Jane-Eyre_1847', 
                        'full_threshold-0%90': 'Bronte_Charlotte_Jane-Eyre_1847', 
                        'full_threshold-0%95': 'Bronte_Charlotte_Jane-Eyre_1847', 
                        'full_authormin': 'Bronte_Charlotte_Jane-Eyre_1847', 
                        'full_simmel-3-10': 'Bronte_Charlotte_Jane-Eyre_1847', 
                        'full_simmel-4-6': 'Shelley_Mary_Frankenstein_1818', 
                        'full_simmel-5-10': 'Shelley_Mary_Valperga_1823',
                        'full_simmel-7-10': 'Doyle_Arthur-Conan_The-Red-Headed-League_1891'
                    }

                    self.examples = {
                        'cosinesim-5000_authormax': 'James_Henry_The-Turn-of-the-Screw_1898', 
                        'cosinedelta-1000_threshold-0%8': 'Kipling_Rudyard_On-the-Strength-of-a-Likeness_1888', 
                        'both_threshold-0%90': 'Dickens_Charles_Barnaby-Rudge_1841', # dot to avoid confusion with 0%95
                        'cosinesim-1000_threshold-0%95': 'Kipling_Rudyard_How-the-Whale-Got-His-Throat_1902', 
                        'eder-2000_authormin': 'Walpole_Horace_Otranto_1764', 
                        'sqeuclidean-500_simmel-3-10': 'James_Henry_What-Maisie-Knew_1897', 
                        'both_simmel-4-6': 'Wells_H-G_Tono-Bungay_1909', 
                        'burrows-500_simmel-5-10': 'Corelli_Marie_The-Sorrows-of-Satan_1895',
                        'eder-5000_simmel-7-10': 'Dickens_Charles_David-Copperfield_1849', 
                    }


                else:

                    self.examples_full = {
                        'full_authormax': 'Buechner_Georg_Lenz_1839', 
                        'full_threshold-0%8': 'Buechner_Georg_Lenz_1839', 
                        'full_threshold-0%90': 'Buechner_Georg_Lenz_1839', 
                        'full_threshold-0%95': 'Buechner_Georg_Lenz_1839', 
                        'full_authormin': 'Buechner_Georg_Lenz_1839', 
                        'full_simmel-3-10': 'Goethe_Johann-Wolfgang_Leiden-des-jungen-Werther_1774', 
                        'full_simmel-4-6': 'Storm_Theodor_Ein-Doppelgaenger_1887', 
                        'full_simmel-5-10': 'Lenz_Jakob_Der-Waldbruder_1776',
                        'full_simmel-7-10': 'Keller_Gottfried_Ursula_1877'
                    }


                    self.examples = {
                        'both_threshold-0%8': 'Gutzkow_Karl_Die-Ritter-vom-Geiste_1850',
                        'manhattan-2000_threshold-0%90': 'Tieck_Ludwig_Die-Vogelscheuche_1834',
                        'cosinedelta-1000_threshold-0%95': 'Sacher-Masoch_Leopold_Venus-im-Pelz_1869',
                        'burrows-5000_authormin': 'Wezel_Johann-Karl_Die-Erziehung-der-Moahi_1777',
                        'minmax-500_authormax': 'Schlaf_Johannes_Fruehling_1896',
                        'sqeuclidean-500_simmel-3-10': 'Saar_Ferdinand_Ninon_1892',
                        'canberra-2000_simmel-4-6': 'Suttner_Bertha-von_Die-Waffen-nieder_1889',
                        'sqeuclidean-5000_simmel-5-10': 'Dronke_Ernst_Polizeigeschichten_1846',
                        'edersimple-2000_simmel-7-10': 'Conrad_Michael-Georg_Was-die-Isar-rauscht_1887',
                    }
                





            else:
                if self.language == 'eng':

                    self.examples_full = {
                        'full_threshold-0%8': 'Bronte_Charlotte_all_1850', 
                        'full_threshold-0%90': 'Bronte_Charlotte_all_1850', 
                        'full_threshold-0%95': 'Bronte_Charlotte_all_1850',  
                        'full_simmel-3-10': 'Bronte_Charlotte_all_1850',  
                        'full_simmel-4-6': 'Shelley_Mary_all_1827', 
                        'full_simmel-5-10': 'Shelley_Mary_all_1827',
                        'full_simmel-7-10': 'Doyle_Arthur-Conan_all_1898'
                    }

                    self.examples = {
                        'cosinedelta-1000_threshold-0%8': 'Kipling_Rudyard_all_1892', 
                        'both_threshold-0%90': 'Kipling_Rudyard_all_1892', # dot to avoid confusion with 0%95
                        'cosinesim-1000_threshold-0%95': 'Kipling_Rudyard_all_1892', 
                        'sqeuclidean-500_simmel-3-10': 'James_Henry_all_1898', 
                        'both_simmel-4-6': 'Wells_H-G_all_1903', 
                        'burrows-500_simmel-5-10': 'Corelli_Marie_all_1895',
                        'eder-5000_simmel-7-10': 'Kipling_Rudyard_all_1892', 
                    }


                else:

                    self.examples_full = {
                        'full_threshold-0%8': 'Buechner_Georg_all_1839', 
                        'full_threshold-0%90': 'Buechner_Georg_all_1839', 
                        'full_threshold-0%95': 'Buechner_Georg_all_1839', 
                        'full_simmel-3-10': 'Goethe_Johann-Wolfgang_all_1799', 
                        'full_simmel-4-6': 'Storm_Theodor_all_1877', 
                        'full_simmel-5-10': 'Lenz_Jakob_all_1776',
                        'full_simmel-7-10': 'Keller_Gottfried_all_1867'
                    }


                    self.examples = {
                        'both_threshold-0%8': 'Gutzkow_Karl_all_1838',
                        'manhattan-2000_threshold-0%90': 'Tieck_Ludwig_all_1813',
                        'cosinedelta-1000_threshold-0%95': 'Sacher-Masoch_Leopold_all_1880',
                        'sqeuclidean-500_simmel-3-10': 'Saar_Ferdinand_all_1891',
                        'canberra-2000_simmel-4-6': 'Suttner_Bertha-von_all_1889',
                        'sqeuclidean-5000_simmel-5-10': 'Dronke_Ernst_all_1846',
                        'edersimple-2000_simmel-7-10': 'Conrad_Michael-Georg_all_1890',
                    }
            


            self.examples.update(self.examples_full)



                
            self.nklist = self.examples.keys()
            self.edgelists = [
                filename for filename in self.edgelists
                if any(substring in filename for substring in self.nklist)
            ]

        elif self.mode == 'run' or self.mode == 'bestparams':
            # intnk_path = os.path.join(self.data_dir, 'analysis', self.language, 'interesting_networks.csv')
            # self.nklist = []
            # with open(intnk_path, 'r') as f:
            #     for line in f:
            #         row = line.rstrip()
            #         self.nklist.append(row)
            # # self.nklist = ['cosinesim-2000_threshold-0%90'] ##############################################

            # # Filter edgelist based on nklist
            # filtered_egelists = []
            # for i in self.nklist:
            #     el_name = f'{i}.csv'
            #     assert el_name in self.edgelists, f'{el_name} not in self.edgelists'
            #     filtered_egelists.append(el_name)
            # self.edgelists = filtered_egelists
            self.edgelists = self.edgelists # use all networks instead of interesting networks

            
        self.nr_mxs = 58
        self.nr_spars = 9
        self.remove_iso = True
        self.index_mapping = pd.read_csv(os.path.join(self.edgelist_dir, 'index-mapping.csv'), header=0, dtype={'new_index': str})
        self.logger.debug(f'Read index mapping from file. "new_index" is str.')



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
        

    def network_from_edgelist(self, file_path, delimiter=',', nodes_as_str=False, print_info=False):
        if self.remove_iso:
                self.logger.debug(f'Isolated nodes are removed from the graph.')

        if 'threshold' in file_path:
                graph = nx.read_weighted_edgelist(file_path, delimiter=delimiter)
        else:
                graph = nx.read_weighted_edgelist(file_path, delimiter=delimiter, create_using=nx.DiGraph())


        if nodes_as_str:
        # Create a new graph with updated node names
            mapping = dict(zip(self.index_mapping['new_index'], self.index_mapping['original_index']))
            graph = nx.relabel_nodes(graph, mapping)

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

        for edgelist in tqdm(self.edgelists):
        # if 'braycurtis-2000' in edgelist:
            if self.language == 'eng' and ('cosinesim-500_simmel-4-6' in edgelist or 'cosinesim-500_simmel-7-10' in edgelist): # no edges
                continue
            print(edgelist)
            network = self.network_from_edgelist(os.path.join(self.edgelist_dir, edgelist))
            if nx.is_directed(network): # nx.connected_components only implemented for undirected
                network = network.to_undirected()
            components = list(nx.connected_components(network))
            component_sizes = sorted([len(component) for component in components])
            all_sizes = ','.join([str(x) for x in component_sizes])

            largest_component = max(components, key=len)
            largest_component_graph = network.subgraph(largest_component)
            aspl = nx.average_shortest_path_length(largest_component_graph)
            diam = nx.diameter(largest_component_graph)
            
            nrs = [edgelist, min(component_sizes), max(component_sizes), len(component_sizes), all_sizes, aspl, diam]
            stats.append(nrs)

        stats = pd.DataFrame(stats, columns=['file_name', 'min_component_size', 'max_component_size', 'nr_components', 'all_size', 'av_shortest_path_length', 'diameter'])
        stats.to_csv(self.stats_path, header=True, index=False)


    def get_network_selfloops(self):
        stats = []
        for edgelist in self.edgelists:
                if self.language == 'eng' and ('cosinesim-500_simmel-4-6' in edgelist or 'cosinesim-500_simmel-7-10' in edgelist): # no edges
                    continue
                print(edgelist)
                network = self.network_from_edgelist(os.path.join(self.edgelist_dir, edgelist))
                if nx.is_directed(network): # nx.connected_components only implemented for undirected
                    network = network.to_undirected()
                components = list(nx.connected_components(network))
                component_sizes = sorted([len(component) for component in components])

                nr_selfloops = nx.number_of_selfloops(network)

                all_sizes = ','.join([str(x) for x in component_sizes])
                nrs = [edgelist, nr_selfloops, len(component_sizes), all_sizes]
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
    def __init__(self, language, output_dir, edgelist_dir='sparsification_edgelists', mode=None, by_author=False):
        super().__init__(language, output_dir=output_dir, edgelist_dir=edgelist_dir, mode=mode, by_author=by_author)
        self.add_subdir('embeddings')


    def load_embeddings(self, file_name):
        '''
        Map embeddings to string node names.
        '''

        inpath = os.path.join(self.subdir, f'{file_name}.embeddings')
        # First line in embedding file contains number of nodes and number of dimensions, there is no header
        # Use the first column with the node ID as index
        df = pd.read_csv(inpath, skiprows=1, header=None, sep=' ', index_col=0, dtype={0: str})


        # Rename columns as col1, col2, ...
        df.columns = [f'col{i}' for i in range(1, len(df.columns) + 1)]

        # Map int IDs to file names
        df = df.merge(self.index_mapping, left_index=True, right_on='new_index', validate='1:1')
        df = df.set_index('original_index')
        df = df.drop('new_index', axis=1)
        df.index = df.index.rename('file_name')

        return df
    

    def get_param_string(self, kwargs):
        d = {key.replace('-', ''): value for key, value in kwargs.items()} # remove '-', such as in 'walk-length', which are required in the s2v main script
        return '_'.join(f'{key}-{value}' for key, value in d.items())
    

    def get_embedding_path(self, edgelist, kwargs):
        if '.csv' in edgelist:
            edgelist = os.path.splitext(edgelist)[0]
        param_string = self.get_param_string(kwargs)
        return os.path.join(self.subdir, f'{edgelist}_{param_string}.embeddings')
    

    def get_all_embedding_paths(self):
        paths = []
        param_combs = self.get_param_combinations()
        for comb in param_combs:
            for edgelist in self.edgelists:
                # if any(substring in edgelist for substring in self.noedges): ########## keep hardcoded matrix names because importing causes trouble with conda environments
                if self.language == 'eng' and ('cosinesim-500_simmel-4-6' in edgelist or 'cosinesim-500_simmel-7-10' in edgelist): # have no edges
                    continue
                
                embedding_path = self.get_embedding_path(edgelist, comb)
                paths.append(embedding_path)
        
        return paths
   

    def generate_paths(self, kwargs={}):
        for edgelist in self.edgelists:
            if self.language == 'eng' and ('cosinesim-500_simmel-4-6' in edgelist or 'cosinesim-500_simmel-7-10' in edgelist):
                continue
            
            embedding_path = self.get_embedding_path(edgelist, kwargs)
            yield (edgelist, embedding_path)


    def check_embedding_dimensions(self):
        # Check if embeddings files have the same nuber of nodes as edgelist files
        embedding_dim_log_path = os.path.join(self.output_dir, 'embedding_dimensions_check.csv')
        embedding_index_log_path = os.path.join(self.output_dir, 'embedding_index_check.csv')
        with open(embedding_dim_log_path, 'w') as f:
            f.write(f'edgelist,embedding_path,nrow_embeddings,nrow_firstline,nnodes_edgelist\n')

        param_combs = self.get_param_combinations()
        for edgelist in self.edgelists:
            edgelist_path = os.path.join(self.edgelist_dir, edgelist)
            graph = nx.read_weighted_edgelist(edgelist_path, delimiter=' ')
            nnodes =  graph.number_of_nodes()
            
            
            for comb in param_combs:
                # if any(substring in edgelist for substring in self.noedges): ########## keep hardcoded matrix names because importing causes trouble with conda environments
                if self.language == 'eng' and ('cosinesim-500_simmel-4-6' in edgelist or 'cosinesim-500_simmel-7-10' in edgelist): # don't have edges
                    continue
                
                embedding_path = self.get_embedding_path(edgelist, comb)

                if os.path.exists(embedding_path): ###########################

                    with open(embedding_path, 'r') as file:
                        first_line = file.readline().strip()
                        nrow_firstline, ndim_firstline = first_line.split()
                        nrow_firstline = int(nrow_firstline)
                        ndim_firstline = int(ndim_firstline)
                    
                        # First line in embedding file contains number of nodes and number of dimensions, there is no header
                        # Use the first column with the node ID as index
                        df = pd.read_csv(embedding_path, skiprows=1, header=None, sep=' ', index_col=0, dtype={0: str})
                        nrow = df.shape[0]
                        ncol = df.shape[1]

                        nrow_firstline_equal = nrow == nrow_firstline
                        ncol_firstline_equal = ncol == ndim_firstline
                        nrow_nnodes_equal = nrow == nnodes

                        if not nrow_nnodes_equal or not nrow_firstline_equal:
                            with open(embedding_dim_log_path, 'a') as f:
                                f.write(f'{edgelist},{embedding_path},{nrow},{nrow_firstline},{nnodes}\n')


                        # Check if the index values are a sequence from 1 to len(df) with no gaps
                        duplicated_index = df.index.duplicated().any()
                        if duplicated_index:
                            with open(embedding_index_log_path, 'a') as f:
                                f.write(f'{embedding_path}\n')



    def check_single_embedding_dimensions(self, edgelist, embedding_path):
        edgelist_path = os.path.join(self.edgelist_dir, edgelist)
        graph = nx.read_weighted_edgelist(edgelist_path, delimiter=' ')
        nnodes =  graph.number_of_nodes()
            
        if os.path.exists(embedding_path):

            with open(embedding_path, 'r') as file:
                first_line = file.readline().strip()
                nrow_firstline, ndim_firstline = first_line.split()
                nrow_firstline = int(nrow_firstline)
                ndim_firstline = int(ndim_firstline)
            
                # First line in embedding file contains number of nodes and number of dimensions, there is no header
                # Use the first column with the node ID as index
                df = pd.read_csv(embedding_path, skiprows=1, header=None, sep=' ', index_col=0, dtype={0: str})
                nrow = df.shape[0]
                ncol = df.shape[1]

                nrow_firstline_equal = nrow == nrow_firstline
                ncol_firstline_equal = ncol == ndim_firstline
                nrow_nnodes_equal = nrow == nnodes

                if not nrow_nnodes_equal or not nrow_firstline_equal:
                    print(f'edgelist {edgelist}\nembedding_path {embedding_path}\nnrow_embeddings {nrow}\nnrow_firstline {nrow_firstline}\nnnodes_edgelist {nnodes}\n')
                else:
                    print(f'embedding_path {embedding_path}: embedding has correct dimensions.')
        


    def create_data(self, kwargs={}):
        for edgelist, embedding_path in self.generate_paths(kwargs):
            if not os.path.exists(embedding_path):
                self.create_embeddings(edgelist, embedding_path, kwargs)
            # else:
            #     print('already exists:', embedding_path)


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


    def count_combinations(self):
        param_comb = self.get_param_combinations()
        nr_param_comb = len(param_comb)
        if self.mode == 'params':
            nr_networks = len(self.examples)
        else:
            nr_networks = len(self.edgelists)
        nr_comb = nr_param_comb * nr_networks
        return nr_comb
    

    def check_embeddings(self):
        # Check nr of files
        nr_comb = self.count_combinations()
        print(f'Expected number of combinations: {nr_comb}')
        nr_embeddings = len(os.listdir(self.subdir))
        print(f'Nr combinations in subdir: {nr_embeddings}')

        # Check if files exist
        all_paths = self.get_all_embedding_paths()
        not_exists_counter = 0
        for path in all_paths:
            if not os.path.exists(path):
                print('Embeddings do not exist:', path)
                not_exists_counter += 1
        print(f'Expected Nr. of embeddings for mode {self.mode}: {nr_embeddings}. Nr. of missing embeddings: {not_exists_counter}')



        self.check_embedding_dimensions()



# for language in ['eng', 'ger']:
#     ns = NetworkStats(language)
#     ns.get_network_stats()
# %%
