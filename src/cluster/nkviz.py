
import pandas as pd
import itertools
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pickle
from copy import deepcopy
import os
import matplotlib.gridspec as gridspec
import textwrap
import time
import random
random.seed(9)


import sys
sys.path.append("..")
from utils import DataHandler
from .cluster_utils import CombinationInfo
import logging
logging.basicConfig(level=logging.DEBUG)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pygraphviz") # Suppress warning: Error: remove_overlap: Graphviz not built with triangulation library

class NkViz(DataHandler):

    def __init__(self, language, network, info):
        super().__init__(language, output_dir='similarity', data_type='png')
        self.noviz_path = self.get_file_path(file_name='log-noviz.txt', subdir=False)
        self.network = network
        self.graph = self.network.graph
        self.info = info
        self.prog = 'neato'
        self.cat_attrs = ['gender', 'author']
        self.add_subdir('nktop')

        self.too_many_edges, edges_info = self.check_nr_edges()
        if self.too_many_edges:
            self.write_noviz(edges_info)
        else:
            self.prepare_graphs_and_plot()


    def write_noviz(self, edges_info):
        # Write combination info that was just run to file
        with open(self.noviz_path, 'a') as f:
            f.write(f'{self.info.as_string()},{edges_info.as_string()}\n')


    def make_figure(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5.5), gridspec_kw={'height_ratios': [7, 0.5]})
        ax1.axis('off')
        ax2.axis('off')
        return fig, ax1, ax2
    

    def visualize(self, pltname, plttitle):
        if not self.too_many_edges:
            start = time.time()
            df = self.prepare_metadata(pltname)
            self.add_nodes(df, plttitle, pltname)

            # Make second visualization for categorical attributes where cluster and attribute are combined into a color
            if (pltname == 'evalviz') and (self.info.attr in self.cat_attrs):
                df = df.drop('color', axis=1)
                df = df.rename(columns={f'{self.info.attr}_cluster_color': 'color'})
                df['shape'] = 'o'
                self.add_nodes(df, plttitle, pltname, use_different_shapes=False, fn_str='combined')
                
            calctime = time.time()-start
            if calctime > 10:
                print(f'{calctime}s to visualize.')


    def prepare_graphs_and_plot(self):
        print(self.info.as_string())
        print(self.info.as_string(omit=["clst_alg_params", "attr"]))
        pkl_path = self.get_file_path(file_name=f'pg-{self.info.as_string(omit=["clst_alg_params", "attr"])}.pkl', subdir=True)

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.graph_con, self.graphs_two, self.nodes_removed, self.nodes_iso, self.pos, self.fig = pickle.load(f)
                self.ax1, self.ax2 = self.fig.get_axes()
                self.ax1.axis('off')
                self.ax2.axis('off')
                self.logger.debug(f'Loaded graphs and plot from file.')

        else:
            self.logger.debug(f'Nr edges below cutoff for {self.info.as_string()}. Making visualization.') ##################3
            self.graph_con, self.graphs_two, self.nodes_removed, self.nodes_iso = self.get_graphs()
            self.pos = self.get_positions()
            self.fig, self.ax1, self.ax2 = self.make_figure()
            self.draw_edges(self.graph_con, self.pos, self.ax1)


            pkl_lst = [self.graph_con, self.graphs_two, self.nodes_removed, self.nodes_iso, self.pos, self.fig]
            try:
                with open(pkl_path, 'wb') as f:
                    pickle.dump(pkl_lst, f)
            except Exception as e:
                if os.path.exists(pkl_path):
                    os.remove(pkl_path)
                print(f"Pickling failed, regenerating edges plot for every visualization.: {e}")

            self.logger.debug(f'Finished preparing viz {self.info.as_string()}')



    def add_nodes(self, df, plttitle, pltname, use_different_shapes=True, fn_str=None):
        # Draw connected components with more than 2 nodes
        df_con = df[~df.index.isin(self.nodes_removed)]
        self.draw_nodes(self.graph_con, self.ax1, df_con, use_different_shapes=use_different_shapes)

        ## Plot removed nodes, if there are any
        if self.nodes_removed:
            # Two nodes
            print('nr graphs two', len(self.graphs_two))
            for curr_g in self.graphs_two:
                curr_nodes = list(curr_g.nodes)
                curr_df_two = df[df.index.isin(curr_nodes)]
                self.draw_nodes(curr_g, self.ax2, curr_df_two, use_different_shapes=use_different_shapes)
                self.draw_edges(curr_g, self.pos, self.ax2) #€#####################

            # Isolated nodes
            df_iso = df[df.index.isin(self.nodes_iso)]
            for shape in df_iso['shape'].unique():
                sdf = df_iso[df_iso['shape'] == shape]
                self.ax2.scatter(sdf['x'], sdf['y'], c=sdf['color'], marker=shape, s=2)


        plt.tight_layout()
    
        if plttitle is not None:
            plt.suptitle(textwrap.fill(plttitle, width=100), fontsize=5)

        if fn_str is None:
            file_name = f'{pltname}_{self.info.as_string()}.{self.data_type}'
        else:
            file_name = f'{pltname}_{self.info.as_string()}_{fn_str}.{self.data_type}'

        self.save_data(data=plt, data_type=self.data_type, subdir=True, file_name=file_name)
   


    def draw_nodes(self, graph, ax, df, use_different_shapes=True):
        # Iterate through shapes because only one shape can be passed at a time, no lists
        if use_different_shapes:
            shapes = df['shape'].unique()
        else:
            shapes = ['o']
            df = df.copy() # Avoid chained assingment warning
            df['shape'] = 'o'

        for shape in shapes:
            sdf = df[df['shape'] == shape]
            nx.draw_networkx_nodes(graph, 
                                self.pos, 
                                ax=ax,
                                nodelist=sdf.index.tolist(), 
                                node_shape=shape,
                                node_color=sdf['color'],
                                node_size=5,
                                edgecolors='black',
                                linewidths=0.2)
            

    def draw_edges(self, graph, pos, ax):
        start = time.time()
        edge_weights = nx.get_edge_attributes(graph, 'weight')
        weights_list = list(edge_weights.values())
        # alpha for opacity
        nx.draw_networkx_edges(graph, 
                               pos, 
                               ax=ax, 
                               edge_color=weights_list, 
                               edge_cmap=plt.cm.get_cmap('gist_yarg'), 
                               arrowsize=2, 
                               width=0.5, 
                               arrows=False, 
                               alpha=0.1) ################## arrows
        ax.grid(False)
        # print(f'{time.time()-start}s to draw edges.')


    def prepare_metadata(self, pltname):
        node_colors = self.info.metadf[f'{self.info.attr}_color'].to_dict()
        attr = self.info.metadf[self.info.attr].to_dict()
        if pltname == 'evalviz':
            node_shapes = self.info.metadf['clst_shape'].to_dict()
        else:
            node_shapes = {key: 'o' for key in node_colors}
        
        df_dict = {'color': node_colors, 'shape': node_shapes, f'{self.info.attr}': attr, 'pos': self.pos}

        # Also visualize combined attribute-cluster color column for categorical attributes
        if (pltname=='evalviz') and (self.info.attr in self.cat_attrs):
            attr_cluster_col = self.info.metadf[f'{self.info.attr}_cluster_color'].to_dict()
            df_dict[f'{self.info.attr}_cluster_color'] = attr_cluster_col

        df = pd.DataFrame(df_dict)

        #df['pos'] = df.index.map(pos)
        df[['x', 'y']] = pd.DataFrame(df['pos'].tolist(), index=df.index)

        return df


    def get_graphs(self):
        # Connected components with 2 nodes
        # nx.connected_components is only implemented for undirected graphs
        if nx.is_directed(self.graph):
            graph = self.graph.to_undirected()
        else:
            graph = deepcopy(self.graph)
        graphs_two = [graph.subgraph(comp).copy() for comp in nx.connected_components(graph) if len(comp) == 2]
        # Extract nodes from the connected components with 2 nodes
        nodes_two = [node for subgraph in graphs_two for node in subgraph.nodes()]

        # Isolated nodes
        nodes_iso = list(nx.isolates(self.graph))
        nodes_removed = nodes_two + nodes_iso
    
        # Main graphs
        graph_con = self.graph.subgraph([node for node in self.graph.nodes if node not in nodes_removed])
        return graph_con, graphs_two, nodes_removed, nodes_iso
    

    def get_positions(self):
        '''
        Calculate node positions.
        Use pygraphviz for layout and NetworkX for visualization.
        If layout programs are used on whole graph, isolated nodes are randomly distributed.
        To adress this, connected (2 nodes that are only connected to each other) and isolated nodes are visualized separately.
        '''
        
        # Calculate node positions for main graph and removed nodes
        nodes_per_line = 50

        start = time.time()

        # Custom grid layout for two-node components and isolated nodes
        row_height = 0.01
        pos_removed = {node: (i % nodes_per_line, -(i // nodes_per_line) * row_height) for i, node in enumerate(self.nodes_removed)}

        pos_con = nx.nx_agraph.graphviz_layout(self.graph_con, self.prog) # dict ##########################

        pos = {**pos_removed, **pos_con}

        # print(f'{time.time()-start}s to get node layout.')
        return pos


    def count_visible_edges(self):
        edges = list(self.network.graph.edges())

        # Count visible edges. If there is an edge from A to B and from B to A, is it counted only once
        if nx.is_directed(self.network.graph):
            unique_edges = set()
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                unique_edges.add(sorted_edge)
            nrvis = len(unique_edges)

        else:
            nrvis = len(edges)

        nr_possible_edges = self.network.mx.mx.shape[0]**2
        ratio_vis = nrvis/nr_possible_edges

        edges_info = CombinationInfo(nr_edges=len(edges), nr_vis_edges=nrvis, ratio_vis_edges=ratio_vis)

        return edges_info
    

    def check_nr_edges(self):
        # Check if the number of edges is too high to make a good plot
        edges_info = self.count_visible_edges()
        threshold = 0.2 # Set by inspecting plots

        if edges_info.ratio_vis_edges > threshold:
            self.logger.debug(f'Nr edges above cutoff for {self.info.as_string()}')
            return True, edges_info
        else:
            return False, edges_info
        

    def set_info(self, info):
        self.info = info



    # def save_graphml(self, node_colors, pos):
    #     # Pos is a dict with format: file_name: (x_position, y_position)
    #     # Graphml can not handle tuples as attributes
    #     # Create a dict for each position, store them as separate attributes
    #     graph = deepcopy(self.graph)
    #     nx.set_node_attributes(graph, values=pos, name='pos')
    #     nx.set_node_attributes(graph, values=node_colors, name='color')
            
    #     self.save_data(data=self.graph, data_type='graphml', subdir=True, file_name=f'nk-{self.info.as_string()}.graphml')
    #     # for node in graph.nodes():
    #     #     attributes = graph.nodes[node]
    #     #     print(f"Node {node} attributes: {attributes}")
        
