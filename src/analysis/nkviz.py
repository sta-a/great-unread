
import pandas as pd
import itertools
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pickle
from copy import deepcopy
import os
import matplotlib.gridspec as gridspec
import time
import random
import textwrap
from typing import List
random.seed(9)


import sys
sys.path.append("..")
from utils import DataHandler
from .analysis_utils import VizBase
from cluster.cluster_utils import CombinationInfo
import logging
logging.basicConfig(level=logging.DEBUG)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pygraphviz") # Suppress warning: Error: remove_overlap: Graphviz not built with triangulation library


class NkViz(VizBase):

    def __init__(self, language, network, info, plttitle, expname):
        self.cmode = 'nk'
        super().__init__(language, self.cmode, info, plttitle, expname)
        self.network = network
        self.graph = self.network.graph
        self.info.metadf.to_csv('nkvizmeta', index=True, header=True) #################3

        # Visualization parameters
        self.prog = 'neato'
        self.markersize = 20
        self.fontsize = 10
        self.global_vmax, self.global_vmin = self.get_cmap_params()

        # Whitespace
        self.ws_left = 0.03
        self.ws_right = 0.97
        self.ws_bottom = 0.05
        self.ws_top = 0.95
        self.ws_wspace = 0.05
        self.ws_hspace = 0.1

        # If graph has too many edges, it cannot be drawn
        self.noviz_path = self.get_file_path(file_name=f'{self.cmode}_log-noviz.txt', subdir=True)
        self.too_many_edges, edges_info = self.check_nr_edges()
        if self.too_many_edges:
            self.write_noviz(edges_info)

        if not self.too_many_edges:
            self.logger.debug(f'Nr edges below cutoff for {self.info.as_string()}. Making visualization.')
            self.get_graphs()
            self.get_positions()
            self.prepare_metadata()


    def get_cmap_params(self):
        '''
        Calculate vmax and vmin for consistent colormap scaling based on edge weights.

        Retrieves edge weights from the graph and determines the maximum (vmax) and minimum (vmin)
        values to ensure a consistent color scale in visualizations using colormaps.
        '''
        weights = list(nx.get_edge_attributes(self.graph, 'weight').values())
        return max(weights), min(weights)


    def write_noviz(self, edges_info):
        # Write combination info that was just run to file
        with open(self.noviz_path, 'a') as f:
            f.write(f'{self.info.as_string()},{edges_info.as_string()}\n')


    def visualize(self, vizname='viz', omit=[]):
        if not self.too_many_edges:
            self.vizpath = self.get_path(name=vizname, omit=omit)
            if not os.path.exists(self.vizpath):
                start = time.time()
                self.logger.debug(f'Nr edges below cutoff for {self.info.as_string()}. Making visualization.')

                self.get_figure()
                self.adjust_subplots()
                self.add_edges()

                self.fill_subplots()

                # self.fig.set_constrained_layout(True)
                self.save_plot(plt)
                plt.show()
                    
                calctime = time.time()-start
                if calctime > 10:
                    print(f'{calctime}s to visualize.')


    def add_edges(self):
        # Main plot
        print('subplots', self.subplots)
        for ix in self.subplots:
            s = time.time()
            self.draw_edges(self.graph_con, self.pos, ix)
            # print(f'{time.time()-s}s to draw edges for one plot.')

        # Two nodes
        if self.nodes_removed:
            for ix in self.subplots:
                if ix is not None:
                    ix = [ix[0]+1, ix[1]]
                    for curr_g in self.graphs_two:
                        self.draw_edges(curr_g, self.pos, ix)


    def add_nodes_to_ax(self, ix, df, color_col, use_different_shapes=False):
        color_col = f'{color_col}_color'
        # Draw connected components with more than 2 nodes
        df_con = df[~df.index.isin(self.nodes_removed)]
        ax = self.get_ax(ix)
        s = time.time()
        self.draw_nodes(self.graph_con, ax, df_con, color_col, use_different_shapes=use_different_shapes)
        # print(f'{time.time()-s}s to draw nodes for one plot.')

        ax = self.axs[ix[0]+1, ix[1]]
        # Isolated nodes
        if self.nodes_removed:
            # Two nodes
            for curr_g in self.graphs_two:
                curr_nodes = list(curr_g.nodes)
                curr_df_two = df[df.index.isin(curr_nodes)]
                self.draw_nodes(curr_g, ax, curr_df_two, color_col, use_different_shapes=use_different_shapes)

            # Isolated nodes
            df_iso = df[df.index.isin(self.nodes_iso)]
            if use_different_shapes:
                for shape in df_iso['clst_shape'].unique():
                    sdf = df_iso[df_iso['clst_shape'] == shape]
                    ax.scatter(sdf['x'], sdf['y'], c=sdf[color_col], marker=shape, s=2)
            else:
                ax.scatter(df_iso['x'], df_iso['y'], c=df_iso[color_col], marker='o', s=2)


    def adjust_subplots(self):
        self.fig.subplots_adjust(
            left=self.ws_left,
            right=self.ws_right,
            bottom=self.ws_bottom,
            top=self.ws_top,
            wspace=self.ws_wspace,
            hspace=self.ws_hspace)  
        

    def get_figure(self):
        # Add column for legends and titles at the end

        if self.is_cat and self.has_special:
            ncol = 4
        else:
            ncol = 3
        
        if self.is_topattr_viz:
            ncol += 1
            if self.is_cat:
                ncol += 1
            if self.by_author:
                ncol += 1

        width_ratios = (ncol-1)*[7] + [1]

        self.fig, self.axs = plt.subplots(4, ncol, figsize=(sum(width_ratios), 11), gridspec_kw={'height_ratios': [7, 0.5, 7, 0.5], 'width_ratios': width_ratios})      

        for row in self.axs: 
            for ax in row:
                ax.axis('off')

        self.attrix = [0,0]
        self.clstix = [0,1]
        self.shapeix = [2,0]

        if self.is_cat and self.has_special:
            self.combix = [2,1]
            self.specix = [0,2]
        if self.is_cat and not self.has_special:
            self.combix = [2,1]
            self.specix = None
        if not self.is_cat and self.has_special:
            self.combix = None
            self.specix = [2,1]
        if not self.is_cat and not self.has_special:
            self.combix = None
            self.specix = None

        if self.is_topattr_viz: # no special
            if not self.is_cat:
                self.first_ext_ix = [2,1]
                self.second_ext_ix = [0,2]
                self.third_ext_ix = [2,2]
                if self.by_author:
                    # Add plots for canon-max and canon-min. 
                    # Only one plot needs to be added because 'author' is not visualized.
                    self.fourth_ext_ix = [0,3]
            else:
                self.first_ext_ix = [0,2]
                self.second_ext_ix = [2,2]  
                self.third_ext_ix = [0,3]
                if self.by_author:
                    self.fourth_ext_ix = [2,3]


        self.subplots = [self.attrix, self.clstix, self.shapeix, self.combix, self.specix]
        if self.is_topattr_viz:
            self.subplots.extend([self.first_ext_ix, self.second_ext_ix, self.third_ext_ix])
            if self.by_author:
                self.subplots.extend([self.fourth_ext_ix])

        

    def add_legends_and_titles(self):

        def make_clst_legend(fig_or_ax, boxx, boxy):
            self.add_legend(
                fig_or_ax=fig_or_ax,
                attr='cluster',
                label='size',
                use_shapes=True,
                loc='upper left',
                boxx=boxx,
                boxy=boxy,
                boxwidth=0.05,
                boxheight=0.1,
                fontsize=self.fontsize,
                markersize=self.markersize)
            
        def make_attr_legend(fig_or_ax, boxx, boxy):
            self.add_legend(
                fig_or_ax=fig_or_ax,
                attr=self.info.attr,
                label='attr',
                use_shapes=False,
                loc='upper left',
                boxx=boxx,
                boxy=boxy,
                boxwidth=0.05,
                boxheight=0.1,
                fontsize=self.fontsize,
                markersize=self.markersize)
        
        
        # Add all extra elements to third column
        make_clst_legend(self.fig, boxx=0.9, boxy=0.9)
        if self.is_cat:
            make_attr_legend(self.fig, boxx=0.9, boxy=0.7)
        if self.needs_cbar:
            self.add_cbar(self.axs[2, -1])


        # Add subtitles to subplots
        self.add_subtitles(self.attrix, self.clstix, self.shapeix, self.combix, self.specix)
        self.add_text(self.axs[3, -1], width=25)
        

    def fill_subplots(self):
        # attr
        self.add_nodes_to_ax(self.attrix, self.df, color_col=self.info.attr, use_different_shapes=False)

        # cluster
        self.add_nodes_to_ax(self.clstix, self.df, color_col='cluster', use_different_shapes=False)

        # attr as color, cluster as shapes
        self.add_nodes_to_ax(self.shapeix, self.df, color_col=self.info.attr, use_different_shapes=True)

        # cluster and attr combined as colors
        if self.is_cat:
            self.add_nodes_to_ax(self.combix, self.df, color_col=f'{self.info.attr}_cluster', use_different_shapes=False)

        # special attribute
        if self.has_special:
            self.add_nodes_to_ax(self.specix, self.df, color_col=self.info.special, use_different_shapes=True)

        
        if self.is_topattr_viz:    
            self.add_nodes_to_ax(self.first_ext_ix, self.df, color_col=self.key_attrs[0], use_different_shapes=True)
            self.add_nodes_to_ax(self.second_ext_ix, self.df, color_col=self.key_attrs[1], use_different_shapes=True)
            self.add_nodes_to_ax(self.third_ext_ix, self.df, color_col=self.key_attrs[2], use_different_shapes=True)
            self.get_ax(self.first_ext_ix).set_title(self.key_attrs[0], fontsize=self.fontsize)
            self.get_ax(self.second_ext_ix).set_title(self.key_attrs[1], fontsize=self.fontsize)
            self.get_ax(self.third_ext_ix).set_title(self.key_attrs[2], fontsize=self.fontsize)

            if self.by_author:
                self.add_nodes_to_ax(self.fourth_ext_ix, self.df, color_col=self.key_attrs[3], use_different_shapes=True)
                self.get_ax(self.fourth_ext_ix).set_title(self.key_attrs[3], fontsize=self.fontsize)


        self.add_legends_and_titles()


    def prepare_metadata(self):
        # Combine positions and metadata
        df = deepcopy(self.info.metadf)
        df['pos'] = df.index.map(self.pos)
        df[['x', 'y']] = pd.DataFrame(df['pos'].tolist(), index=df.index)
        self.df = df


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
                                node_size=self.markersize,
                                edgecolors='black',
                                linewidths=0.2)
                 

    def draw_edges(self, graph, pos, ix):
        if ix is not None:
            ax = self.get_ax(ix)
            start = time.time()

            edge_weights = nx.get_edge_attributes(graph, 'weight')
            edge_color = list(edge_weights.values())
            nx.draw_networkx_edges(graph, 
                                pos, 
                                ax=ax, 
                                edge_color=edge_color, 
                                edge_cmap=plt.cm.get_cmap('gist_yarg'),
                                edge_vmax=self.global_vmax,
                                edge_vmin=self.global_vmin,
                                arrowsize=2, 
                                width=0.5, 
                                arrows=False, 
                                alpha=0.3) # alpha for opacity

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
        self.graphs_two = [graph.subgraph(comp).copy() for comp in nx.connected_components(graph) if len(comp) == 2]
        # Extract nodes from the connected components with 2 nodes
        nodes_two = [node for subgraph in self.graphs_two for node in subgraph.nodes()]

        # Isolated nodes
        self.nodes_iso = list(nx.isolates(self.graph))
        self.nodes_removed = nodes_two + self.nodes_iso
        print('nodes removed: ', len(self.nodes_removed))
    
        # Main graphs
        self.graph_con = self.graph.subgraph([node for node in self.graph.nodes if node not in self.nodes_removed])
    

    def get_positions(self):
        '''
        Calculate node positions.
        Use pygraphviz for layout and NetworkX for visualization.
        If layout programs are used on whole graph, isolated nodes are randomly distributed.
        To adress this, connected (2 nodes that are only connected to each other) and isolated nodes are visualized separately.
        '''
        if not hasattr(self, 'pos'):
            # Calculate node positions for main graph and removed nodes
            nodes_per_line = 40

            # Custom grid layout for two-node components and isolated nodes
            row_height = 0.5
            pos_removed = {node: (i % nodes_per_line, -(i // nodes_per_line) * row_height) for i, node in enumerate(self.nodes_removed)}

            pos_con = nx.nx_agraph.graphviz_layout(self.graph_con, self.prog)

            self.pos = {**pos_removed, **pos_con}
        else:
            print('self.pos already exists.')


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


    def save_graphml(self):
        # Pos is a dict with format: file_name: (x_position, y_position)
        # Graphml can not handle tuples as attributes
        # Create a dict for each position, store them as separate attributes
        graph = deepcopy(self.graph)

        for node, row in self.df.iterrows():
            graph.nodes[node]['x'] = row['x']
            graph.nodes[node]['y'] = row['y']
            # graph.nodes[node]['cluster'] = row['cluster']
            # graph.nodes[node]['attr'] = row[self.info.attr]
            # graph.nodes[node]['attr_color'] = row[f'{self.info.attr}_color'] # no tuples

            # if f'{self.info.attr}_cluster_color' in df.columns:
            #     graph.nodes[node]['combined_color'] = df[f'{self.info.attr}_cluster_color']

            
        # for node in graph.nodes():
        #     attributes = graph.nodes[node]
        #     print(f"Node {node} attributes: {attributes}")
    
        return graph
    


class NkVizAttr(NkViz):
    def __init__(self, language, network, info, plttitle, expname):
        super().__init__(language, network, info, plttitle, expname)
        self.data_type = 'png'
        # if self.expname == 'attrviz':
        #     self.nrow = 2
        #     self.ncol = 4
        # else:
        #     self.nrow = 2
        #     self.ncol = 2
        self.nrow = 7
        self.ncol = 14
        self.markersize = 10
        self.fontsize = 6

        # Whitespace
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0
        self.ws_hspace = 0


    def create_logfile(self, all_cols, nfields):
        df = pd.DataFrame({'feature': all_cols})

        df['cmode'] = self.cmode
        df['mxname'] = self.info.mxname
        df['sparsmode'] = self.info.sparsmode
        df['distinctive'] = ''

        # Nr of plot that contains feature
        viznr_values = []
        for i in range(len(df)):
            viznr_values.append(i // nfields)
        df['viznr'] = viznr_values

        df = df[['cmode', 'mxname', 'sparsmode', 'viznr', 'feature', 'distinctive']]
        self.save_data(data=df, subdir=True, file_name='visual-assessment.csv', data_type='csv')


    def visualize(self, vizname='viz'): # vizname for compatibility
        all_cols = self.get_feature_columns(self.info.metadf)
        nfields = self.nrow * self.ncol # fields per plot
        nplots = len(all_cols)
        print('nplots', nplots)
        nfig = nplots // nfields 
        if nplots % nfields != 0:
            nfig += 1
        # self.create_logfile(all_cols, nfields)

        ix = 0
        for i in range(nfig):
            # If ix + nfields > len(cols), no error is raised because Python allows out-of-bound slicing
            self.cols = all_cols[ix: ix + (nfields)]
            ix += nfields
            omit=['clst_alg_params']
            if self.nrow*self.ncol != 1:
                omit.append('attr')
            else:
                self.info.drop('attr')
                self.info.add('attr', deepcopy(self.cols[0]))
            s = time.time()

            if nfig == 1:
                vizname = vizname
            else:
                vizname = f'{vizname}{i}'
            super().visualize(vizname=vizname, omit=omit)
            print(f'{time.time()-s}s to make one fig.')


    def get_figure(self):
        height = self.nrow*3
        width = self.ncol*3
        if self.nrow == 1 and self.ncol == 1:
            self.fig, self.axs = plt.subplots(figsize=(width, height))
            self.axs = np.reshape(self.axs, (1, 1))  # Convert single Axes object to a 2D numpy array
        else:
            self.fig, self.axs = plt.subplots(self.nrow, self.ncol, figsize=(width, height))


        # min_xlim = min(self.pos[node][0] for node in self.graph_con.nodes()) - 0.1
        # max_xlim = max(self.pos[node][0] for node in self.graph_con.nodes()) + 0.1
        # min_ylim = min(self.pos[node][1] for node in self.graph_con.nodes()) - 0.1
        # max_ylim = max(self.pos[node][1] for node in self.graph_con.nodes()) + 0.1
        # self.axs[i, j].set_xlim([min_xlim, max_xlim])
        # self.axs[i, j].set_ylim([min_ylim, max_ylim])

        # for i in range(self.nrow):
        #     for j in range(self.ncol):
        #         self.axs[i, j].set_xticks([])
        #         self.axs[i, j].set_yticks([])
        #         self.axs[i, j].set_xticklabels([])
        #         self.axs[i, j].set_yticklabels([])
        #         self.axs[i, j].set_aspect('equal')  # Set equal aspect ratio for squares

        for ax in self.axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal') 

            ax.spines['top'].set_visible(False)  # Hide the top spine
            ax.spines['right'].set_visible(False)  # Hide the right spine
            ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
            ax.spines['left'].set_visible(False)  # Hide the left spine

        plt.tight_layout()


    def add_edges(self):
        souter = time.time()

        for i in range(self.nrow):
            for j in range(self.ncol):
                index = i * self.ncol + j
                if index < len(self.cols):
                    self.draw_edges(self.graph_con, self.pos, [i, j])

        print(f'{time.time()-souter}s to draw edges for all plot.')


    def add_nodes_to_ax(self, ix, df, color_col, use_different_shapes=False):
        color_col = f'{color_col}_color'
        # Draw connected components with more than 2 nodes
        df_con = df[~df.index.isin(self.nodes_removed)]
        ax = self.get_ax(ix)
        s = time.time()
        self.draw_nodes(self.graph_con, ax, df_con, color_col, use_different_shapes=use_different_shapes)
        # print(f'{time.time()-s}s to add nodes to one plot.')


    def fill_subplots(self):
        for i in range(self.nrow):
            for j in range(self.ncol):
                index = i * self.ncol + j
                if index < len(self.cols):
                    self.add_nodes_to_ax([i, j], self.df, color_col=f'{deepcopy(self.cols[index])}', use_different_shapes=False)
                    # self.axs[i, j].set_title(f'{deepcopy(self.cols[index])}', fontsize=self.fontsize)
                    self.axs[i, j].text(0.05, 0.05, f'{deepcopy(self.cols[index])}', transform=self.axs[i, j].transAxes, bbox=dict(facecolor='white', alpha=0.5))
                    self.axs[i, j].axis('off')

