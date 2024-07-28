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
        results = {2: [], 10: []}
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


            if nx.is_directed(network): # nx.connected_components only implemented for undirected
                network = network.to_undirected()
            if nx.is_directed(network):
                components = sorted(list(nx.weakly_connected_components(network)))
            else:
                 components = sorted(list(nx.connected_components(network)))


            for compsize in [2, 10]:
                component_results = []
                comp_sizes = []
                total_weighted_ac_year = 0
                total_weighted_ac_canon = 0
                total_weighted_ac_author = 0
                total_nodes_in_components = 0
                for component in components:
                    if len(component) < compsize: # Not interesting for smaller components
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
                        ac_year = 1
                    else:
                        ac_year = nx.numeric_assortativity_coefficient(subgraph, 'year')
                    if sub_df['canon'].nunique() == 1:
                        ac_canon = 1
                    else:
                        ac_canon = nx.numeric_assortativity_coefficient(subgraph, 'canon')


                    if not self.by_author:
                        for node in sub_df.index:
                            subgraph.nodes[node]['author'] = df.loc[node, 'author']
                        if sub_df['author'].nunique() == 1:
                            ac_author = 1
                        else:
                            ac_author = nx.attribute_assortativity_coefficient(subgraph, 'author')
                    else:
                        ac_author = np.nan

                    component_results.append([ac_year, ac_canon, ac_author])
                    total_weighted_ac_year += len(component) * ac_year
                    total_weighted_ac_canon+= len(component) * ac_canon
                    total_weighted_ac_author += len(component) * ac_author
                    total_nodes_in_components += len(component)


    
                if component_results:
                    component_results_df = pd.DataFrame(component_results, columns=['ac_year', 'ac_canon', 'ac_author'])
                    aggregated_results = {
                        'mxname': mxname.split('.')[0].replace('sparsmx-', ''),
                        'weighted_avg_ac_year': total_weighted_ac_year/ total_nodes_in_components if total_nodes_in_components > 0 else None,
                        'weighted_avg_ac_canon': total_weighted_ac_canon/ total_nodes_in_components if total_nodes_in_components > 0 else None,
                        'weighted_avg_ac_author': total_weighted_ac_author/ total_nodes_in_components if total_nodes_in_components > 0 else None,
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
                        'comp_sizes': ','.join(comp_sizes),
                    }

                    results[compsize].append(aggregated_results)

                results_df = pd.DataFrame(results[compsize])
                self.save_data(file_name=f'author-year-canon_compsize-{compsize}', data=results_df, sep='\t')



    def iterate_mxs_cat(self):
        results = {2: [], 10: []}
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


            for compsize in [2, 10]:
                component_results = []
                comp_sizes = []
                total_weighted_ac_gender = 0
                total_nodes_in_components = 0
                for component in components:
                    if len(component) < compsize:
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
                        ac_gender = 1
                    else:
                        ac_gender = nx.attribute_assortativity_coefficient(subgraph, 'gender')

                    component_results.append([ac_gender]) 
                    total_weighted_ac_gender += len(component) * ac_gender
                    total_nodes_in_components += len(component)

                if component_results:
                    component_results_df = pd.DataFrame(component_results, columns=['ac_gender'])

                    aggregated_results = {
                        'mxname': mxname.split('.')[0].replace('sparsmx-', ''),
                        'weighted_avg_ac_gender': total_weighted_ac_gender / total_nodes_in_components if total_nodes_in_components > 0 else None,
                        'avg_ac_gender': component_results_df['ac_gender'].mean() if not component_results_df['ac_gender'].isnull().all() else None,
                        'min_ac_gender': component_results_df['ac_gender'].min() if not component_results_df['ac_gender'].isnull().all() else None,
                        'max_ac_gender': component_results_df['ac_gender'].max() if not component_results_df['ac_gender'].isnull().all() else None,
                        'all_ac_gender': component_results_df['ac_gender'].tolist(),
                        'comp_sizes': ','.join(comp_sizes),
                    }
                    results[compsize].append(aggregated_results)

                results_df = pd.DataFrame(results[compsize])
                self.save_data(file_name=f'gender_compsize-{compsize}', data=results_df, sep='\t')


for language in ['eng', 'ger']:
    m  = Assort(language, by_author=False)
    m.iterate_mxs_cont()
    m.iterate_mxs_cat()


# %%
