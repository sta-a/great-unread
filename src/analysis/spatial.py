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

            

    def iterate_mxs(self):
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
            components = list(nx.connected_components(network))

            # Calculate Moran's I for each component individually 
            component_results = []
            for component in components:
                if len(component) < 10: # Moran's I for smaller values are not interesting
                    continue
                
                subgraph = network.subgraph(component)
                sub_df = df.loc[list(component)]

                self.check_df_and_graph_nodes(df=sub_df, network=subgraph)
                assert set(subgraph.nodes) == set(sub_df.index)


                isolated_nodes_subgraph = list(nx.isolates(subgraph))
                network.remove_nodes_from(isolated_nodes_subgraph)
                for i in isolated_nodes_subgraph:
                    print('iso node in subgraph', i)
                self.check_df_and_graph_nodes(df=sub_df, network=subgraph)
                assert set(subgraph.nodes) == set(sub_df.index)

                # plt.figure(figsize=(8, 6))
                # nx.draw_networkx(subgraph, with_labels=False, node_color='lightblue', edge_color='gray', node_size=500, font_size=15)
                # plt.show()
                
                adj_matrix = nx.to_numpy_array(subgraph, nodelist=sub_df.index.tolist())
                w = weights.util.full2W(adj_matrix)

                if any(len(neighbors) == 0 for neighbors in w.neighbors.values()):
                    isolated_in_component = [node for node, neighbors in w.neighbors.items() if len(neighbors) == 0]
                    print(f'Isolated nodes within the component: {isolated_in_component}')
                    continue  # Skip this component since it has isolated nodes

                moran_year = esda.moran.Moran(sub_df['year'], w)
                moran_canon = esda.moran.Moran(sub_df['canon'], w)
                jc = esda.Join_Counts(df['gender'], w)

                component_results.append([
                    moran_year.I, moran_year.p_norm,
                    moran_canon.I, moran_canon.p_norm,
                    jc.bb, jc.p_sim_bb,
                    jc.bw, jc.p_sim_bw,
                    jc.ww]) # p_sim_ww not implemented

            if component_results:
                component_results_df = pd.DataFrame(component_results, columns=[
                    'moran_I_year', 'moran_pval_year',
                    'moran_I_canon', 'moran_pval_canon'
                    'n_bb_joins', 'bb_join_pval',
                    'n_bw_joins', 'bw_join_pval',
                    'n_ww_joins'
                ])

                # Aggregate results
                aggregated_results = {
                    'mxname': mxname.split('.')[0].replace('sparsmx-', ''),
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
                }

                results.append(aggregated_results)

        results_df = pd.DataFrame(results)
        self.save_data(file_name='moran', data=results_df)


m  = Autocorr('eng', True)
m.iterate_mxs()
# %%
