# %%
'''
Activate env networkclone
'''
import pandas as pd
import os
import sys
sys.path.append("..")
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysal.explore import esda
from pysal.lib import weights
from utils import DataHandler
from cluster.network import NXNetwork
from cluster.combinations import InfoHandler



class Autocorr(DataHandler):
    '''
    Calculate the spatial autocorrelation on networks for continuous attributes with Moran's I and for categorical attributes with join counts.
    https://pysal.org/esda/generated/esda.Join_Counts.html
    https://en.wikipedia.org/wiki/Join_count_statistic
    '''
    def __init__(self, language, by_author=False):
        super().__init__(language=language, output_dir='moran', data_type='csv')
        self.by_author = by_author
        self.ih = InfoHandler(language=language, add_color=False, cmode=None, by_author=by_author)
        self.mxdir = os.path.join(self.ih.output_dir, 'sparsification')
        self.ndist = 58
        self.nspars = 7 if self.by_author else 9


    def load_mxnames(self):
        mxs = [filename for filename in os.listdir(self.mxdir) if filename.startswith('sparsmx')]
        if self.by_author:
            noedges_sparsmethods = ['authormin', 'authormax'] # these distance matrices have no edges if author-based
            mxs = [filename for filename in mxs if all(substr not in filename for substr in noedges_sparsmethods)]
        mxs = sorted(mxs)
        assert len(mxs) == (self.ndist * self.nspars)
        return mxs

    def check_df_and_graph_nodes(self, df, network):
        # Check if the graph labels and the DataFrame index are the same
        graph_nodes = set(network.nodes)
        df_index = set(df.index)
        
        if graph_nodes != df_index:
            only_in_graph = graph_nodes - df_index
            only_in_df = df_index - graph_nodes
            
            print('Nodes only in graph:', only_in_graph)
            print('Indices only in DataFrame:', only_in_df)
        else:
            assert graph_nodes == df_index

            

    def iterate_mxs_cont(self):
        '''
        If by_author = False and language = eng, division errors occur. 
        These are probably due to the df containing only one value in the canon or year column.
        This is typically caused if the df contains only texts by the same author.
        '''
        results = []
        mxs = self.load_mxnames()
        for mxname in mxs:
            print(mxname)
            df = deepcopy(self.ih.metadf)
            network = NXNetwork(self.language, path=os.path.join(self.mxdir, mxname))
            network = network.graph

            # Remove isolated nodes
            isolated_nodes = list(nx.isolates(network))
            network.remove_nodes_from(isolated_nodes)
            df = df.drop(index=isolated_nodes, errors='raise')

            '''
            If network is not converted to undirected, islands (nodes with no neighbours) appear.
            These are probably nodes with incoming but no outgoing edges.
            Convert to undirected, edge weights are not that important, position in the network is important.
            '''
            if nx.is_directed(network): # nx.connected_components only implemented for undirected
                network = network.to_undirected()
            components = sorted(list(nx.connected_components(network)))

            # Calculate Moran's I for each component individually 
            component_results = []
            comp_sizes = []
            for component in components:
                if len(component) < 10: # Moran's I for smaller values are not interesting
                    continue

                comp_sizes.append(str(len(component)))
                
                subgraph = network.subgraph(component)
                sub_df = df.loc[list(component)]
                self.check_df_and_graph_nodes(df=sub_df, network=subgraph)
                assert set(subgraph.nodes) == set(sub_df.index)

                print(sub_df['year'])
                print(sub_df['canon'])

  

                adj_matrix = nx.to_numpy_array(subgraph, nodelist=sub_df.index.tolist())
                w = weights.util.full2W(adj_matrix)
                moran_year = esda.moran.Moran(sub_df['year'], w)
                moran_canon = esda.moran.Moran(sub_df['canon'], w)
    
                component_results.append([moran_year.I, moran_year.p_norm, moran_canon.I, moran_canon.p_norm])
 

            if component_results:
                component_results_df = pd.DataFrame(component_results, columns=[
                    'moran_I_year', 'moran_pval_year',
                    'moran_I_canon', 'moran_pval_canon'])
                aggregated_results = {
                    'mxname': mxname.split('.')[0].replace('sparsmx-', ''),
                    'all_moran_I_year': component_results_df['moran_I_year'].tolist(),
                    'all_moran_I_canon': component_results_df['moran_I_canon'].tolist(),
                    'all_moran_pval_year': component_results_df['moran_pval_year'].tolist(),
                    'all_moran_pval_canon': component_results_df['moran_pval_canon'].tolist(),
                    'avg_moran_I_year': component_results_df['moran_I_year'].mean(),
                    'avg_moran_pval_year': component_results_df['moran_pval_year'].mean(),
                    'avg_moran_I_canon': component_results_df['moran_I_canon'].mean(),
                    'avg_moran_pval_canon': component_results_df['moran_pval_canon'].mean(),
                    'min_moran_I_year': component_results_df['moran_I_year'].min(),
                    'max_moran_I_year': component_results_df['moran_I_year'].max(),
                    'min_moran_pval_year': component_results_df['moran_pval_year'].min(),
                    'max_moran_pval_year': component_results_df['moran_pval_year'].max(),
                    'min_moran_I_canon': component_results_df['moran_I_canon'].min(),
                    'max_moran_I_canon': component_results_df['moran_I_canon'].max(),
                    'min_moran_pval_canon': component_results_df['moran_pval_canon'].min(),
                    'max_moran_pval_canon': component_results_df['moran_pval_canon'].max(),
                    'comp_sizes': (',').join(comp_sizes),
                }

                results.append(aggregated_results)

        results_df = pd.DataFrame(results)
        self.save_data(file_name='cont', data=results_df, sep='\t')



    def iterate_mxs_cat(self):
        results = []
        mxs = self.load_mxnames()
        for mxname in mxs:
            print(mxname)
            df = deepcopy(self.ih.metadf)
            network = NXNetwork(self.language, path=os.path.join(self.mxdir, mxname))
            network = network.graph

            # Keep only nodes with gender 0 or 1
            df = df[df['gender'].isin([0, 1])]
            filtered_nodes = set(df.index)
            network = network.subgraph(filtered_nodes).copy()  # Make a mutable copy  to avoid frozen graph
            self.check_df_and_graph_nodes(df=df, network=network)
            assert set(network.nodes) == set(df.index)

            # Remove isolated nodes
            isolated_nodes = list(nx.isolates(network))
            network.remove_nodes_from(isolated_nodes)
            df = df.drop(index=isolated_nodes, errors='raise')

            '''
            If network is not converted to undirected, islands (nodes with no neighbours) appear.
            These are probably nodes with incoming but no outgoing edges.
            Convert to undirected, edge weights are not that important, position in the network is important.
            '''
            if nx.is_directed(network): # nx.connected_components only implemented for undirected
                network = network.to_undirected()
            components = sorted(list(nx.connected_components(network)))

            # Calculate Moran's I for each component individually 
            component_results = []
            comp_sizes = []
            for component in components:
                if len(component) < 10: # Moran's I for smaller values are not interesting
                    continue

                comp_sizes.append(str(len(component)))
                
                subgraph = network.subgraph(component)
                sub_df = df.loc[list(component)]
                self.check_df_and_graph_nodes(df=sub_df, network=subgraph)
                assert set(subgraph.nodes) == set(sub_df.index)


                adj_matrix = nx.to_numpy_array(subgraph, nodelist=sub_df.index.tolist())
                w = weights.util.full2W(adj_matrix)

                # plt.figure(figsize=(8, 6))
                # nx.draw_networkx(subgraph, with_labels=False, node_color='lightblue', edge_color='gray', node_size=500, font_size=15)
                # plt.show()
                adj_matrix_gender = nx.to_numpy_array(subgraph, nodelist=sub_df.index.tolist())
                w_gender = weights.util.full2W(adj_matrix_gender)
                jc = esda.Join_Counts(sub_df['gender'], w_gender)

                component_results.append([jc.bb, jc.p_sim_bb, jc.bw, jc.p_sim_bw, jc.ww]) # p_sim_ww not implemented
 

            if component_results:
                component_results_df = pd.DataFrame(component_results, columns=[
                    'n_bb_joins', 'bb_join_pval',
                    'n_bw_joins', 'bw_join_pval',
                    'n_ww_joins'
                ])
                aggregated_results = {
                    'mxname': mxname.split('.')[0].replace('sparsmx-', ''),
                    'all_n_bb_joins': component_results_df['n_bb_joins'].tolist(),
                    'all_bb_join_pval': component_results_df['bb_join_pval'].tolist(),
                    'all_n_bw_joins': component_results_df['n_bw_joins'].tolist(),
                    'all_bw_join_pval': component_results_df['bw_join_pval'].tolist(),
                    'all_n_ww_joins': component_results_df['n_ww_joins'].tolist(),
                    'avg_n_bb_joins': component_results_df['n_bb_joins'].mean(),
                    'avg_bb_join_pval': component_results_df['bb_join_pval'].mean(),
                    'avg_n_bw_joins': component_results_df['n_bw_joins'].mean(),
                    'avg_bw_join_pval': component_results_df['bw_join_pval'].mean(),
                    'avg_n_ww_joins': component_results_df['n_ww_joins'].mean(),
                    'min_n_bb_joins': component_results_df['n_bb_joins'].min(),
                    'max_n_bb_joins': component_results_df['n_bb_joins'].max(),
                    'min_bb_join_pval': component_results_df['bb_join_pval'].min(),
                    'max_bb_join_pval': component_results_df['bb_join_pval'].max(),
                    'min_n_bw_joins': component_results_df['n_bw_joins'].min(),
                    'max_n_bw_joins': component_results_df['n_bw_joins'].max(),
                    'min_bw_join_pval': component_results_df['bw_join_pval'].min(),
                    'max_bw_join_pval': component_results_df['bw_join_pval'].max(),
                    'min_n_ww_joins': component_results_df['n_ww_joins'].min(),
                    'max_n_ww_joins': component_results_df['n_ww_joins'].max(),
                    'comp_sizes': (',').join(comp_sizes),
                }

                results.append(aggregated_results)

        results_df = pd.DataFrame(results)
        self.save_data(file_name='cat', data=results_df, sep='\t')



class Assort(DataHandler):
    '''
    Calculate the assortativity coefficients on networks.
    '''
    def __init__(self, language, by_author=False):
        super().__init__(language=language, output_dir='assort', data_type='csv')
        self.by_author = by_author
        self.ih = InfoHandler(language=language, add_color=False, cmode=None, by_author=by_author)
        self.mxdir = os.path.join(self.ih.output_dir, 'sparsification')
        self.ndist = 58
        self.nspars = 7 if self.by_author else 9


    def load_mxnames(self):
        mxs = [filename for filename in os.listdir(self.mxdir) if filename.startswith('sparsmx')]
        if self.by_author:
            noedges_sparsmethods = ['authormin', 'authormax'] # these distance matrices have no edges if author-based
            mxs = [filename for filename in mxs if all(substr not in filename for substr in noedges_sparsmethods)]
        mxs = sorted(mxs)
        assert len(mxs) == (self.ndist * self.nspars)
        return mxs

    def check_df_and_graph_nodes(self, df, network):
        # Check if the graph labels and the DataFrame index are the same
        graph_nodes = set(network.nodes)
        df_index = set(df.index)
        
        if graph_nodes != df_index:
            only_in_graph = graph_nodes - df_index
            only_in_df = df_index - graph_nodes
            
            print('Nodes only in graph:', only_in_graph)
            print('Indices only in DataFrame:', only_in_df)
        else:
            assert graph_nodes == df_index

            

    def iterate_mxs_cont(self):
        '''
        If by_author = False and language = eng, division errors occur. 
        These are probably due to the df containing only one value in the canon or year column.
        This is typically caused if the df contains only texts by the same author.
        '''
        results = []
        mxs = self.load_mxnames()
        for mxname in mxs:
            if 'argamonquadratic-5000_simmel-3-10' not in mxname:
                continue ###############################
            print(mxname)
            df = deepcopy(self.ih.metadf)
            network = NXNetwork(self.language, path=os.path.join(self.mxdir, mxname))
            network = network.graph

            # Remove isolated nodes
            isolated_nodes = list(nx.isolates(network))
            network.remove_nodes_from(isolated_nodes)
            df = df.drop(index=isolated_nodes, errors='raise')


            if nx.is_directed(network): # nx.connected_components only implemented for undirected
                network = network.to_undirected()
            if nx.is_directed(network):
                components = sorted(list(nx.weakly_connected_components(network)))
            else:
                 components = sorted(list(nx.connected_components(network)))

            component_results = []
            comp_sizes = []
            for component in components:
                if len(component) < 10: # Not interesting for smaller components
                    continue

                comp_sizes.append(str(len(component)))
                
                subgraph = network.subgraph(component)
                sub_df = df.loc[list(component)]
                self.check_df_and_graph_nodes(df=sub_df, network=subgraph)
                assert set(subgraph.nodes) == set(sub_df.index)

                # Add node attributes
                for node in sub_df.index:
                    subgraph.nodes[node]['canon'] = df.loc[node, 'canon']
                    subgraph.nodes[node]['year'] = df.loc[node, 'year']

                # Errors if all attributes have the same value
                if sub_df['year'].nunique() == 1:
                    ac_year = np.nan
                else:
                    ac_year = nx.numeric_assortativity_coefficient(subgraph, 'year')
                if sub_df['canon'].nunique() == 1:
                    ac_canon = np.nan
                else:
                    ac_canon = nx.numeric_assortativity_coefficient(subgraph, 'canon')


                if not self.by_author:
                    for node in sub_df.index:
                        subgraph.nodes[node]['author'] = df.loc[node, 'author']
                    if sub_df['author'].nunique() == 1:
                        ac_author = np.nan
                    else:
                        ac_author = nx.attribute_assortativity_coefficient(subgraph, 'author')
                else:
                    ac_author = np.nan

                component_results.append([ac_year, ac_canon, ac_author])
 
            if component_results:
                component_results_df = pd.DataFrame(component_results, columns=['ac_year', 'ac_canon', 'ac_author'])
                aggregated_results = {
                    'mxname': mxname.split('.')[0].replace('sparsmx-', ''),
                    'comp_sizes': ','.join(comp_sizes),
                    'avg_ac_year': component_results_df['ac_year'].mean() if not component_results_df['ac_year'].isnull().all() else None,
                    'avg_ac_canon': component_results_df['ac_canon'].mean() if not component_results_df['ac_canon'].isnull().all() else None,
                    'avg_ac_author': component_results_df['ac_author'].mean() if not component_results_df['ac_author'].isnull().all() else None,
                    'min_ac_year': component_results_df['ac_year'].min() if not component_results_df['ac_year'].isnull().all() else None,
                    'min_ac_canon': component_results_df['ac_canon'].min() if not component_results_df['ac_canon'].isnull().all() else None,
                    'min_ac_author': component_results_df['ac_author'].min() if not component_results_df['ac_author'].isnull().all() else None,
                    'max_ac_year': component_results_df['ac_year'].max() if not component_results_df['ac_year'].isnull().all() else None,
                    'max_ac_canon': component_results_df['ac_canon'].max() if not component_results_df['ac_canon'].isnull().all() else None,
                    'max_ac_author': component_results_df['ac_author'].max() if not component_results_df['ac_author'].isnull().all() else None,
                    'all_ac_year': component_results_df['ac_year'].tolist(),
                    'all_ac_canon': component_results_df['ac_canon'].tolist(),
                    'all_ac_author': component_results_df['ac_author'].tolist(),
                }

                results.append(aggregated_results)

        results_df = pd.DataFrame(results)
        self.save_data(file_name='cont', data=results_df, sep='\t')



    def iterate_mxs_cat(self):
        results = []
        mxs = self.load_mxnames()
        for mxname in mxs:
            print(mxname)
            df = deepcopy(self.ih.metadf)
            network = NXNetwork(self.language, path=os.path.join(self.mxdir, mxname))
            network = network.graph

            # Keep only nodes with gender 0 or 1
            df = df[df['gender'].isin([0, 1])]
            filtered_nodes = set(df.index)
            network = network.subgraph(filtered_nodes).copy()  # Make a mutable copy  to avoid frozen graph
            self.check_df_and_graph_nodes(df=df, network=network)
            assert set(network.nodes) == set(df.index)

            # Remove isolated nodes
            isolated_nodes = list(nx.isolates(network))
            network.remove_nodes_from(isolated_nodes)
            df = df.drop(index=isolated_nodes, errors='raise')

  
            if nx.is_directed(network): # nx.connected_components only implemented for undirected
                network = network.to_undirected()
            if nx.is_directed(network):
                components = sorted(list(nx.weakly_connected_components(network)))
            else:
                 components = sorted(list(nx.connected_components(network)))

            # Calculate Moran's I for each component individually 
            component_results = []
            comp_sizes = []
            for component in components:
                if len(component) < 10: # Moran's I for smaller values are not interesting
                    continue

                comp_sizes.append(str(len(component)))
                
                subgraph = network.subgraph(component)
                sub_df = df.loc[list(component)]
                self.check_df_and_graph_nodes(df=sub_df, network=subgraph)
                assert set(subgraph.nodes) == set(sub_df.index)


                # Add node attributes
                for node in sub_df.index:
                    subgraph.nodes[node]['gender'] = df.loc[node, 'gender']

                if sub_df['gender'].nunique() == 1:
                    ac_gender = np.nan
                else:
                    ac_gender = nx.attribute_assortativity_coefficient(subgraph, 'gender')

                component_results.append([ac_gender]) 

            if component_results:
                component_results_df = pd.DataFrame(component_results, columns=['ac_gender'])

                aggregated_results = {
                    'mxname': mxname.split('.')[0].replace('sparsmx-', ''),
                    'comp_sizes': ','.join(comp_sizes),
                    'avg_ac_gender': component_results_df['ac_gender'].mean() if not component_results_df['ac_gender'].isnull().all() else None,
                    'min_ac_gender': component_results_df['ac_gender'].min() if not component_results_df['ac_gender'].isnull().all() else None,
                    'max_ac_gender': component_results_df['ac_gender'].max() if not component_results_df['ac_gender'].isnull().all() else None,
                    'all_ac_gender': component_results_df['ac_gender'].tolist(),
                }
                results.append(aggregated_results)

        results_df = pd.DataFrame(results)
        self.save_data(file_name='cat', data=results_df, sep='\t')


for language in ['eng', 'ger']:
    m  = Assort(language, by_author=False)
    m.iterate_mxs_cont()
    m.iterate_mxs_cat()

