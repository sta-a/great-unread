
import pandas as pd
import itertools
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pickle
from copy import deepcopy
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
import os
from itertools import product
import matplotlib.gridspec as gridspec
import textwrap
import time
import random
random.seed(9)

from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform

from IPython.display import Image, display

import sys
sys.path.append("..")
from utils import DataHandler
from .create import SimMx
import logging
from .cluster_utils import MetadataHandler
logging.basicConfig(level=logging.DEBUG)

class MxReorder():
    '''Sort row and column indices so that clusters are visible in heatmap.'''

    ORDERS = ['fn', 'olo']

    def __init__(self, language, mx, info):
        self.language = language
        self.mx = mx
        self.info = info
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)


    def order(self):
        order_methods = {
            'fn': self.order_fn,
            'olo': self.order_olo,
            'noattr': self.order_noattr,
            'continuous': self.order_cont,
        }

        if self.info.order in order_methods:
            ordmx = order_methods[self.info.order]()
            if isinstance(ordmx, SimMx):
                ordmx = ordmx.mx
        else:
            raise ValueError(f"Invalid order value: {self.info.order}")

        assert self.mx.mx.shape == ordmx.shape
        assert self.mx.mx.equals(self.mx.mx.T)
        assert ordmx.index.equals(ordmx.columns), 'Index and columns of ordmx must be equal.'

        return ordmx
    

    def order_noattr(self):
        # Order mx according to olo without first sorting by an attribute
        # Add attribute column with constant value
        # Olo is applied to every value of the attribute separately, and only once here because there is only one value
        self.info.metadf[self.info.attr] = 1
        ordmx = self.order_olo()
        return ordmx

    
    def order_cont(self):
        df = self.info.metadf.copy(deep=True)
        file_names = df.sort_values(by=self.info.attr).index.tolist()
        return self.mx.mx.loc[file_names, file_names]
    

    def order_fn(self):
        # Sort rows and columns of each cluster (respectively attribute value) according to file name, which starts with the name of the author
        ordmxs = []

        # Get index labels belonging to the current cluster
        for cluster in self.info.metadf[self.info.attr].unique():
            file_names = self.info.metadf[self.info.metadf[self.info.attr] == cluster].index.tolist()

            df = self.mx.mx.loc[:, file_names].sort_index(axis=1)
            ordmxs.append(df)
        ordmx = pd.concat(ordmxs, axis=1)

        ordmxs = []
        for cluster in self.info.metadf[self.info.attr].unique():
            file_names = self.info.metadf[self.info.metadf[self.info.attr] == cluster].index.tolist()
            df = ordmx.loc[file_names, :].sort_index(axis=0)
            ordmxs.append(df)
        ordmx = pd.concat(ordmxs, axis=0)
        return ordmx

    
    def order_olo(self):
        ordered_fns = []
        # Get unique cluster lables, sorted from rarest to most common label
        unique_clust = self.info.metadf[self.info.attr].value_counts().sort_values().index.tolist()

        # Iterate over unique attribute values
        for cluster in unique_clust:

            # Extract file names for the current cluster
            file_names = self.info.metadf[self.info.metadf[self.info.attr] == cluster].index.tolist()

            # If 1 or 2 elements in cluster, order doesn't matter
            if len(file_names) <=2:
                ordered_fns.extend(file_names)

            # Get OLO for current cluster
            else:
                # Subset the similarity matrix for the current cluster
                cmx = self.mx.dmx.loc[file_names, file_names]
                sq_cmx = squareform(cmx)
                
                cluster_linkage = linkage(sq_cmx, method='average')
                
                order = leaves_list(optimal_leaf_ordering(cluster_linkage, sq_cmx))
                # Map integer indices to string indices
                ordered_fn = cmx.index[order].tolist()
                ordered_fns.extend(ordered_fn)

        # Check that there are no duplicated values
        assert len(set(ordered_fns)) == len(ordered_fns)
        ordmx = self.mx.mx.loc[ordered_fns, ordered_fns]
        ordmx = SimMx(self.language, name='olo', mx=ordmx, normalized=True, is_sim=True, is_directed = self.mx.is_directed, is_condensed=False)

        nr_texts = DataHandler(self.language).nr_texts
        assert (ordmx.mx.shape[0] == nr_texts) and (ordmx.mx.shape[1] == nr_texts) 

        # Reorder the final matrix based on the optimal order of clusters ############################
        return ordmx


class MxViz(DataHandler):
    def __init__(self, language, mx, info):
        super().__init__(language, output_dir='similarity', data_type='png')
        self.mx = mx
        self.info = info
        self.n_jobs = -1

        self.colname = f'{self.info.attr}_color'
        self.add_subdir('mxviz')


    def set_info(self, info):
        # Set metadf from outside class
        self.info = info
        self.colname = f'{self.info.attr}_color'


    def visualize(self, pltname, plttitle=None):
        kwargs = {'aspect': 'auto'}
        if pltname == 'evalviz':
            fig = plt.figure(constrained_layout=False, figsize=(10, 10))
            ax1 = fig.add_subplot(2, 2, 1, **kwargs)
            ax2 = fig.add_subplot(2, 2, 2, **kwargs)
            ax3 = fig.add_subplot(2, 2, 3, projection='3d', **kwargs)
            ax4 = fig.add_subplot(2, 2, 4, projection='3d', **kwargs)

        else:
            fig = plt.figure(constrained_layout=False, figsize=(20, 10))
            gs = fig.add_gridspec(2,4)
            ax1 = fig.add_subplot(gs[0, 0], **kwargs)
            ax2 = fig.add_subplot(gs[0, 1], **kwargs)
            ax3 = fig.add_subplot(gs[1, 0], projection='3d', **kwargs)
            ax4 = fig.add_subplot(gs[1, 1], projection='3d', **kwargs)
            ax5 = fig.add_subplot(gs[:, 2:], **kwargs)
            self.draw_heatmap(ax5)
            
        self.draw_mds(pltname, ax1, ax2, ax3, ax4)
        fig.suptitle(textwrap.fill(plttitle, width=100), fontsize=5)
        self.save_data(data=fig, subdir=True, file_name=f'{pltname}-{self.info.as_string()}.{self.data_type}', plt_kwargs={'dpi': 300})
        plt.close()
        return None


    def draw_mds(self, pltname, ax1, ax2, ax3, ax4):
            # Store layouts because it takes a lot of time to calculate them
            pkl_path = self.get_file_path(file_name=f'mds-{self.mx.name}.pkl', subdir=True) 
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    df = pickle.load(f)
 
            else:
                # Apply classical MDS
                mds_2d = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto', n_jobs=self.n_jobs, random_state=8)
                X_mds_2d = mds_2d.fit_transform(self.mx.dmx)

                # Apply non-metric MDS
                nonmetric_mds_2d = MDS(n_components=2, metric=False, dissimilarity='precomputed', normalized_stress='auto', n_jobs=self.n_jobs, random_state=8)
                X_nonmetric_mds_2d = nonmetric_mds_2d.fit_transform(self.mx.dmx)

                # Apply classical MDS in 3D
                mds_3d = MDS(n_components=3, dissimilarity='precomputed', normalized_stress='auto', n_jobs=self.n_jobs, random_state=8)
                X_mds_3d = mds_3d.fit_transform(self.mx.dmx)

                # Apply non-metric MDS in 3D
                nonmetric_mds_3d = MDS(n_components=3, metric=False, dissimilarity='precomputed', normalized_stress='auto', n_jobs=self.n_jobs, random_state=8)
                X_nonmetric_mds_3d = nonmetric_mds_3d.fit_transform(self.mx.dmx)


                df = pd.DataFrame({
                    'X_mds_2d_0': X_mds_2d[:, 0],
                    'X_mds_2d_1': X_mds_2d[:, 1],
                    'X_nonmetric_mds_2d_0': X_nonmetric_mds_2d[:, 0],
                    'X_nonmetric_mds_2d_1': X_nonmetric_mds_2d[:, 1],
                    'X_mds_3d_0': X_mds_3d[:, 0],
                    'X_mds_3d_1': X_mds_3d[:, 1],
                    'X_mds_3d_2': X_mds_3d[:, 2],
                    'X_nonmetric_mds_3d_0': X_nonmetric_mds_3d[:, 0],
                    'X_nonmetric_mds_3d_1': X_nonmetric_mds_3d[:, 1],
                    'X_nonmetric_mds_3d_2': X_nonmetric_mds_3d[:, 2],
                    })

                with open(pkl_path, 'wb') as f:
                    pickle.dump(df, f)

            if self.info.attr == 'noattr':
                df = df.assign(color=['blue']*len(self.mx.dmx))
            else:
                df = df.assign(color=self.info.metadf.loc[self.mx.dmx.index, self.colname].values)
            df = df.assign(shape=['o']*len(self.mx.dmx))
            if pltname == 'evalviz':
                df = df.assign(shape=self.info.metadf.loc[self.mx.dmx.index, 'clst_shape'].values)

            for shape in df['shape'].unique():
                sdf = df[df['shape'] == shape]
                kwargs = {'c': sdf['color'], 'marker': shape, 's': 10}
                ax1.scatter(sdf['X_mds_2d_0'], sdf['X_mds_2d_1'], **kwargs)
                ax2.scatter(sdf['X_nonmetric_mds_2d_0'], sdf['X_nonmetric_mds_2d_1'], **kwargs)
                ax3.scatter(sdf['X_mds_3d_0'], sdf['X_mds_3d_1'], sdf['X_mds_3d_2'], **kwargs)
                ax4.scatter(sdf['X_nonmetric_mds_3d_0'], sdf['X_nonmetric_mds_3d_1'], sdf['X_nonmetric_mds_3d_2'], **kwargs)

            ax1.set_title('Classical MDS (2D)')
            ax2.set_title('Non-metric MDS (2D)')
            ax3.set_title('Classical MDS (3D)')
            ax4.set_title('Non-metric MDS (3D)')


    def draw_heatmap(self, ax5):
        # Draw heatmap
        ordmx = MxReorder(self.language, self.mx, self.info).order()

        # hot_r, viridis, plasma, inferno
        # ordmx = np.triu(ordmx) ####################
        im = ax5.imshow(ordmx, cmap='coolwarm', interpolation='nearest')
        ax5.axis('off')  # Remove the axis/grid

        # Add a color bar to the heatmap for better understanding of the similarity values
        cbar = plt.colorbar(im, ax=ax5, fraction=0.05, pad=0.1)



class NkViz(DataHandler):
    PROGS = ['sfdp', 'neato', 'kk'] # dot (for directed graphs), circo,  'twopi', 'osage', fdp

    def __init__(self, language, network, info):
        super().__init__(language, output_dir='similarity', data_type='png')
        self.network = network
        self.graph = self.network.graph
        self.info = info
        self.prog = self.info.prog
        self.cat_attrs = ['gender', 'author']

        self.add_subdir('nkviz')
        self.logger.info(f'Drawing graph for {self.info.as_string()}')


    def set_info(self, info):
        self.info = info


    def draw_network_with_shapes(self, graph, pos, ax, df):
        # Iterate through shapes because only one shape can be passed at a time, no lists
        for shape in df['shape'].unique():
            sdf = df[df['shape'] == shape]
            nx.draw_networkx_nodes(graph, 
                                pos, 
                                ax=ax,
                                nodelist=sdf.index.tolist(), 
                                node_shape=shape,
                                node_color=sdf['color'],
                                node_size=5,
                                edgecolors='black',
                                linewidths=0.2)
            
        # Draw edges
        edge_weights = nx.get_edge_attributes(graph, 'weight')
        weights_list = list(edge_weights.values())
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=weights_list, edge_cmap=plt.cm.get_cmap('gist_yarg'), arrowsize=2, width=0.5, arrows=False) ################## arrows
        ax.grid(False)


    def draw_network_no_shapes(self, graph, pos, ax, df):
        nx.draw_networkx_nodes(graph, 
                            pos, 
                            ax=ax,
                            nodelist=df.index.tolist(), 
                            node_shape='o',
                            node_color=df['color'],
                            node_size=5,
                            edgecolors='black',
                            linewidths=0.2)
            
        # Draw edges
        edge_weights = nx.get_edge_attributes(graph, 'weight')
        weights_list = list(edge_weights.values())
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=weights_list, edge_cmap=plt.cm.get_cmap('gist_yarg'), arrowsize=2, width=0.5, arrows=False) ################## arrows
        ax.grid(False)
                

    def count_visible_edges(self):
        edges = list(self.network.graph.edges())
        edges_dict = {'nr_edges': len(edges)}

        # Count visible edges. If there is an edge from A to B and from B to A, is it counted only once
        if nx.is_directed(self.network.graph):
            unique_edges = set()
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                unique_edges.add(sorted_edge)
            edges_dict['nr_vis_edges'] = len(unique_edges)

        else:
            edges_dict['nr_vis_edges'] = len(edges)
        return edges_dict
    

    def prepare_metadata(self, pltname):
        ## Prepare metadata
        node_colors = self.info.metadf[f'{self.info.attr}_color'].to_dict()
        attr = self.info.metadf[self.info.attr].to_dict()
        if pltname == 'evalviz':
            node_shapes = self.info.metadf['clst_shape'].to_dict()
        else:
            node_shapes = {key: 'o' for key in node_colors}
        
        df_dict = {'color': node_colors, 'shape': node_shapes, f'{self.info.attr}': attr}

        # Consider combined attribute-cluster column for categorical attributes
        if (pltname=='evalviz') and (self.info.attr in self.cat_attrs):
            attr_cluster_col = self.info.metadf[f'{self.info.attr}_cluster_color'].to_dict()
            df_dict[f'{self.info.attr}_cluster_color'] = attr_cluster_col

        df = pd.DataFrame(df_dict)
        return df


    def get_graphs(self, df):
        # Connected components with 2 nodes
        # nx.connected_components is only implemented for undirected graphs
        if nx.is_directed(self.graph):
            graph = self.graph.to_undirected()
        else:
            graph = deepcopy(self.graph)
        graphs_two = [graph.subgraph(comp).copy() for comp in nx.connected_components(graph) if len(comp) == 2]
        # Extract nodes from the connected components with 2 nodes
        nodes_two = [node for subgraph in graphs_two for node in subgraph.nodes()]

        # Isolated nodes, sorted by attribute
        df = df.sort_values(by=self.info.attr)
        nodes_iso = list(nx.isolates(self.graph))
        nodes_iso = sorted(nodes_iso, key=lambda x: df.index.get_loc(x))
        nodes_removed = nodes_two + nodes_iso
    
        # Main graphs
        graph_con = self.graph.subgraph([node for node in self.graph.nodes if node not in nodes_removed])
        return graph_con, graphs_two, nodes_removed, nodes_iso
    

    def get_positions(self, df, graph_con, nodes_removed):
        nodes_per_line = 50
        pkl_path = self.get_file_path(file_name=f'pos-{self.info.as_string(omit=["attr"])}.pkl', subdir=True)
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                pos = pickle.load(f)
        else:
            start = time.time()

            # Custom grid layout for two-node components and isolated nodes
            row_height = 0.1
            pos_removed = {node: (i % nodes_per_line, -(i // nodes_per_line) * row_height) for i, node in enumerate(nodes_removed)}

            if self.prog == 'kk':
                pos_con = nx.kamada_kawai_layout(graph_con)
                pos_con = {k: tuple(v) for k,v in pos_con.items()}
            else:
                pos_con = nx.nx_agraph.graphviz_layout(graph_con, self.prog) # dict ##########################

            pos = {**pos_removed, **pos_con}
            print(f'{time.time()-start}s to get pgv layout.')
            with open(pkl_path, 'wb') as f:
                pickle.dump(pos, f)

        df['pos'] = df.index.map(pos)
        df[['x', 'y']] = pd.DataFrame(df['pos'].tolist(), index=df.index)
        return df, pos


    def make_plots(self, df, graph_con, graphs_two, nodes_removed, nodes_iso, pos, plttitle, pltname, pltmethod, fn_str=None):
        df_con = df[~df.index.isin(nodes_removed)]
        fig = plt.subplots(figsize=(5, 7)) # 8, 8.5
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
        ax1 = plt.subplot(gs[0])
        ax1.axis('off')
        pltmethod(graph_con, pos, ax1, df_con)

        ## Plot removed nodes, if there are any
        if nodes_removed:
            # nr_lines = (len(nodes_removed)- 1) // nodes_per_line
            # scaling_param = 0.1

            ax2 = plt.subplot(gs[1])
            ax2.axis('off')
            for curr_g in graphs_two:
                curr_nodes = list(curr_g.nodes)
                curr_df_two = df[df.index.isin(curr_nodes)]
                pltmethod(curr_g, pos, ax2, curr_df_two)

            df_iso = df[df.index.isin(nodes_iso)]
            for shape in df_iso['shape'].unique():
                sdf = df_iso[df_iso['shape'] == shape]
                ax2.scatter(sdf['x'], sdf['y'], c=sdf['color'], marker=shape, s=2)


            plt.tight_layout()
        
            if plttitle is not None:
                plt.suptitle(textwrap.fill(plttitle, width=100), fontsize=5)

            if fn_str is None:
                file_name = f'{pltname}_{self.info.as_string()}.{self.data_type}'
            else:
                file_name = f'{pltname}_{self.info.as_string()}_{fn_str}.{self.data_type}'


            self.save_data(data=plt, data_type=self.data_type, subdir=True, file_name=file_name)
            plt.show()
            plt.close()
            

    def visualize(self, pltname, plttitle=None):
        # Check if the number of edges is too high to make a good plot
        vizdict = self.count_visible_edges()
        nr_possible_edges = self.network.mx.mx.shape[0]**2
        share_vis = vizdict['nr_vis_edges']/nr_possible_edges
        cutoff = 0.2 # Set by inspecting plots
        if share_vis > cutoff:
            vizdict['viz_time'] = 'noviz'
            self.logger.warning(f'Nr edges above cutoff for {self.info.as_string()}')
            return vizdict
        else:
            self.logger.warning(f'Nr edges below cutoff for {self.info.as_string()}. Making visualization.')
            funcstart = time.time()
            # Positions for connected nodes
            # Use pygraphviz for node positions, nx for visualizatoin
            # Store layouts because it takes a lot of time to calculate them
            # If layout programs are used on whole graph, isolated nodes are randomly distributed
            # Visualize connected and isolated nodes separately

            df = self.prepare_metadata(pltname)
            graph_con, graphs_two, nodes_removed, nodes_iso = self.get_graphs(df)
            df, pos = self.get_positions(df, graph_con, nodes_removed)
            self.make_plots(df, graph_con, graphs_two, nodes_removed, nodes_iso, pos, plttitle, pltname, pltmethod=self.draw_network_with_shapes)


            # Make second visualization for categorical attributes where cluster and attribute are combined into a color
            if (pltname == 'evalviz') and (self.info.attr in self.cat_attrs):
                print(self.info.attr, f'{self.info.attr}_cluster_color')
                df = df.drop('color', axis=1)
                df = df.rename(columns={f'{self.info.attr}_cluster_color': 'color'})
                df['shape'] = 'o'
                self.make_plots(df, graph_con, graphs_two, nodes_removed, nodes_iso, pos, plttitle, pltname, pltmethod=self.draw_network_no_shapes, fn_str='combined')


            viztime = time.time()-funcstart
            print(f'{viztime}s to run whole visualization.')
            self.logger.info(f'Finished visualization {self.info.as_string()}')
            vizdict['viz_time'] = viztime
            return vizdict



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