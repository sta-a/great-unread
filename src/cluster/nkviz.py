
import pandas as pd
import itertools
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pickle
from copy import deepcopy
import os
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import time
import random
from typing import List
random.seed(9)


import sys
sys.path.append("..")
from utils import DataHandler
from .cluster_utils import CombinationInfo, Colors, VizBase
import logging
logging.basicConfig(level=logging.DEBUG)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pygraphviz") # Suppress warning: Error: remove_overlap: Graphviz not built with triangulation library

class NkViz(VizBase):

    def __init__(self, language, network, info, plttitle):
        self.cmode = 'nk'
        super().__init__(language, self.cmode, info, plttitle)
        self.network = network
        self.graph = self.network.graph
        self.prog = 'neato'
        self.noviz_path = self.get_file_path(file_name=f'{self.cmode}_log-noviz.txt', subdir=True)

        self.too_many_edges, edges_info = self.check_nr_edges()
        if self.too_many_edges:
            self.write_noviz(edges_info)


    def write_noviz(self, edges_info):
        # Write combination info that was just run to file
        with open(self.noviz_path, 'a') as f:
            f.write(f'{self.info.as_string()},{edges_info.as_string()}\n')


    def get_figure(self):
        self.fig, self.axs = plt.subplots(4, 2, figsize=(10, 11), gridspec_kw={'height_ratios': [7, 0.5, 7, 0.5]})
        # Turn off box around axis
        for row in self.axs: 
            for ax in row:
                ax.axis('off')

        self.main_plots = [self.axs[0, 0], self.axs[0, 1], self.axs[2, 0]]
        self.small_plots = [self.axs[1, 0], self.axs[1, 1], self.axs[3, 0]]
        if self.info.attr in self.cat_attrs:
            self.main_plots.append(self.axs[2, 1])
            self.small_plots.append(self.axs[3, 1])


    # def make_small_figure(self):
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5.5), gridspec_kw={'height_ratios': [7, 0.5]})
    #     ax1.axis('off')
    #     ax2.axis('off')
    #     return fig, ax1, ax2
        

    def visualize(self):
        if not self.too_many_edges:
            start = time.time()
            self.logger.debug(f'Nr edges below cutoff for {self.info.as_string()}. Making visualization.')
            self.graph_con, self.nodes_removed, self.nodes_iso = self.get_graphs()
            self.pos = self.get_positions()

            self.get_figure()
            for ax in self.main_plots:
                self.draw_edges(self.graph_con, self.pos, ax)

            self.df = self.prepare_metadata()
            self.fill_subplots()
                
            calctime = time.time()-start
            if calctime > 10:
                print(f'{calctime}s to visualize.')
    

    # def fig_to_pkl(self, fig, pkl_path):
    #     # Save a fig
    #     try:
    #         with open(pkl_path, 'wb') as f:
    #             pickle.dump(fig, f)
    #     except Exception as e:
    #         if os.path.exists(pkl_path):
    #             os.remove(pkl_path)


    # def fig_from_pkl(self, pkl_path):
    #     # Load a figure with two axes
    #     if os.path.exists(pkl_path):
    #         with open(pkl_path, 'rb') as f:
    #             fig = pickle.load(f)
    #             ax1, ax2 = fig.get_axes()
    #             ax1.axis('off')
    #             ax2.axis('off')


    # def make_small_plots(self):
    #     for name in ['cluster', 'attr']:
    #         self.make_small_plot(name)

    # def make_small_plot(self, name):
    #     pkl_path = self.get_path(name, omit=[], data_type='pkl')
    #     if not os.path.exists(pkl_path):
    #         fig, ax1, ax2 = self.make_small_figure()
    #         self.add_nodes_to_ax(ax1, ax2, self.df, color_col=f'{name}_color', use_different_shapes=False)
    #         self.fig_to_pkl(fig, pkl_path)
    #         fig_path = self.get_path(name, omit=[], data_type=self.data_type)
    #         self.save_plot(data=fig, file_path=fig_path)


    def add_nodes_to_ax(self, ax1, ax2, df, color_col, use_different_shapes=True):
        color_col = f'{color_col}_color'
        # Draw connected components with more than 2 nodes
        df_con = df[~df.index.isin(self.nodes_removed)]
        self.draw_nodes(self.graph_con, ax1, df_con, color_col, use_different_shapes=use_different_shapes)

        # Isolated nodes
        if self.nodes_removed:
            df_iso = df[df.index.isin(self.nodes_iso)]
            for shape in df_iso['clst_shape'].unique():
                sdf = df_iso[df_iso['clst_shape'] == shape]
                ax2.scatter(sdf['x'], sdf['y'], c=sdf[color_col], marker=shape, s=2)


    def fill_subplots(self):
        # attr
        self.add_nodes_to_ax(self.axs[0, 0], self.axs[1, 0], self.df, color_col=self.info.attr, use_different_shapes=False)
        if self.info.attr in self.cat_attrs:
            self.cat_legend(self.axs[0, 0], self.info.attr, label='attr', loc='upper left')
        else:
            self.add_cbar(self.axs[0, 0])

        # cluster
        self.add_nodes_to_ax(self.axs[0, 1], self.axs[1, 1], self.df, color_col='cluster', use_different_shapes=False)
        self.cat_legend(self.axs[0, 1], 'cluster', label='size')

        # Switch first two plots
        # cluster
        # self.add_nodes_to_ax(self.axs[0, 0], self.axs[1, 0], self.df, color_col='cluster', use_different_shapes=False)
        # self.axs[0, 0].set_title('Cluster', fontsize=self.fontsize)
        # self.cat_legend(self.axs[0, 0], 'cluster', label='size')

        # # attr
        # self.add_nodes_to_ax(self.axs[0, 1], self.axs[1, 1], self.df, color_col=self.info.attr, use_different_shapes=False)
        # self.axs[0, 1].set_title('Attribute', fontsize=self.fontsize)
        # if self.info.attr in self.cat_attrs:
        #     self.cat_legend(self.axs[0, 1], self.info.attr, label='attr', loc='upper left')
        # else:
        #     self.add_cbar(self.axs[0, 1])

        # attr as color, cluster as shapes
        self.add_nodes_to_ax(self.axs[2, 0], self.axs[3, 0], self.df, color_col=self.info.attr, use_different_shapes=True)

        # cluster and attr combined as colors
        if self.info.attr in self.cat_attrs:
            self.add_nodes_to_ax(self.axs[2, 1], self.axs[3, 1], self.df, color_col=f'{self.info.attr}_cluster', use_different_shapes=False)

        self.add_subplot_titles(attrax=self.axs[0, 0], clstax=self.axs[0, 1], shapeax=self.axs[2, 0], combax=self.axs[2, 1])
        self.manage_big_figure()
        fig_path = self.get_path('big', omit=[], data_type=self.data_type)
        # self.save_plot(plt, file_name=None, file_path=fig_path)
        # plt.close()
        plt.show()
        


    def prepare_metadata(self):
        df = deepcopy(self.info.metadf)
        df['pos'] = df.index.map(self.pos)
        df[['x', 'y']] = pd.DataFrame(df['pos'].tolist(), index=df.index)

        return df


    def draw_nodes(self, graph, ax, df, color_col, use_different_shapes=True):
        # Iterate through shapes because only one shape can be passed at a time, no lists
        if use_different_shapes:
            shapes = df['clst_shape'].unique()
        else:
            shapes = ['o']
            df = df.copy() # Avoid chained assingment warning
            df['clst_shape'] = 'o'

        for shape in shapes:
            sdf = df[df['clst_shape'] == shape]
            nx.draw_networkx_nodes(graph,
                                self.pos, 
                                ax=ax,
                                nodelist=sdf.index.tolist(), 
                                node_shape=shape,
                                node_color=sdf[color_col],
                                node_size=6,
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
                               alpha=0.3) ################## arrows
        ax.grid(False)
        if time.time()-start > 10:
            print(f'{time.time()-start}s to draw edges.')


    def get_graphs(self):
        # Connected components with 2 nodes
        # nx.connected_components is only implemented for undirected graphs
        if nx.is_directed(self.graph):
            graph = self.graph.to_undirected()
        else:
            graph = deepcopy(self.graph)

        # Isolated nodes
        nodes_iso = list(nx.isolates(self.graph))
        nodes_removed = nodes_iso
    
        # Main graphs
        graph_con = self.graph.subgraph([node for node in self.graph.nodes if node not in nodes_removed])
        return graph_con, nodes_removed, nodes_iso
    

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
        
