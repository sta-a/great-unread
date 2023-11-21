
import pandas as pd
import itertools
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pickle
from copy import deepcopy
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import os
from itertools import product
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

    def __init__(self, language, mx, info, metadf):
        self.language = language
        self.mx = mx
        self.info = info
        self.attr = self.info.attr
        self.metadf = metadf
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)


    def order(self):
        order_methods = {
            'fn': self.order_fn,
            'olo': self.order_olo,
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

    def order_fn(self):
        # Sort rows and columns of each cluster (respectively attribute value) according to file name, which starts with the name of the author
        ordmxs = []

        # Get index labels belonging to the current cluster
        for cluster in self.metadf[self.attr].unique():
            file_names = self.metadf[self.metadf[self.attr] == cluster].index.tolist()

            df = self.mx.mx.loc[:, file_names].sort_index(axis=1)
            ordmxs.append(df)
        ordmx = pd.concat(ordmxs, axis=1)

        ordmxs = []
        for cluster in self.metadf[self.attr].unique():
            file_names = self.metadf[self.metadf[self.attr] == cluster].index.tolist()
            df = ordmx.loc[file_names, :].sort_index(axis=0)
            ordmxs.append(df)
        ordmx = pd.concat(ordmxs, axis=0)
        return ordmx

    
    def order_olo(self):
        ordered_fns = []
        # Get unique cluster lables, sorted from rarest to most common label
        unique_clust = self.metadf[self.attr].value_counts().sort_values().index.tolist()

        # Iterate over unique attribute values
        for cluster in unique_clust:

            # Extract file names for the current cluster
            file_names = self.metadf[self.metadf[self.attr] == cluster].index.tolist()

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

        self.logger.info(f'OLO matrix reorder.')
        return ordmx


class MxViz(DataHandler):
    def __init__(self, language, mx, info):
        super().__init__(language, output_dir='similarity', data_type='png')
        self.mx = mx
        self.info = info
        self.attr=self.info.attr
        self.n_jobs = -1

        self.metadf = None
        self.add_subdir('mxviz')

    def load_metadata(self):
        mh = MetadataHandler(language = self.language, attr=self.attr)
        metadf = mh.get_metadata()
        metadf = mh.add_color(metadf, self.attr)
        self.metadf = metadf

    def set_metadf(self, metadf):
        # Set metadf from outside class
        self.metadf = metadf

    def visualize(self, plttitle):
        if isinstance(plttitle, pd.DataFrame):
            plttitle = ', '.join([f'{col}: {val}' for col, val in plttitle.iloc[0].items()])
        self.draw_heatmap(plttitle)
        self.draw_mds(plttitle)


    def draw_mds(self, plttitle):
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

        # Visualize results in a single plot with four subplots
        fig = plt.figure(figsize=(16, 12))

        # 2D Classical MDS
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(X_mds_2d[:, 0], X_mds_2d[:, 1], c=self.metadf.loc[self.mx.dmx.index, 'color'])
        ax1.set_title('Classical MDS (2D)')

        # 2D Non-metric MDS
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(X_nonmetric_mds_2d[:, 0], X_nonmetric_mds_2d[:, 1], c=self.metadf.loc[self.mx.dmx.index, 'color'])
        ax2.set_title('Non-metric MDS (2D)')

        # 3D Classical MDS
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.scatter(X_mds_3d[:, 0], X_mds_3d[:, 1], X_mds_3d[:, 2], c=self.metadf.loc[self.mx.dmx.index, 'color'])
        ax3.set_title('Classical MDS (3D)')

        # 3D Non-metric MDS
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.scatter(X_nonmetric_mds_3d[:, 0], X_nonmetric_mds_3d[:, 1], X_nonmetric_mds_3d[:, 2], c=self.metadf.loc[self.mx.dmx.index, 'color'])
        ax4.set_title('Non-metric MDS (3D)')

        # Set super title for the entire plot
        fig.suptitle(plttitle)

        # Save the plot
        self.save_data(data=fig, subdir=True, file_name=f'mds-{self.info.as_string()}.{self.data_type}')

        plt.show()


    def draw_heatmap(self, plttitle):
        # Draw heatmap
        ordmx = MxReorder(self.language, self.mx, self.info, self.metadf).order()
 
        # hot_r, viridis, plasma, inferno
        # ordmx = np.triu(ordmx) ####################
        plt.imshow(ordmx, cmap='plasma', interpolation='nearest')
        plt.axis('off')  # Remove the axis/grid

        # Add a color bar to the heatmap for better understanding of the similarity values
        plt.colorbar()

        # Add axis labels and title (optional)
        # plt.xlabel('Data Points')
        # plt.ylabel('Data Points')

        plt.title(plttitle, fontsize=8)

        self.save_data(data=plt, subdir=True, file_name=f'heatmap-{self.info.as_string()}.{self.data_type}')
        plt.close()
    


class NkViz(DataHandler):
    # PROGS = ['fdp', 'dot', 'neato', 'sfdp', 'circo', 'twopi', 'osage']
    PROGS = ['fdp']

    def __init__(self, language, network, info):
        super().__init__(language, output_dir='similarity', data_type='svg')
        self.network = network
        self.graph = self.network.graph
        self.info = info
        self.attr=self.info.attr
        self.prog = self.info.prog

        self.add_subdir('nkviz')
        self.logger.info(f'Drawing graph for {self.info.as_string()}')

    def set_metadf(self, metadf):
        self.metadf = metadf
        assert 'color' in self.metadf.columns
     

    def save_graphml(self, node_colors, pos):
        # Pos is a dict with format: file_name: (x_position, y_position)
        # Graphml can not handle tuples as attributes
        # Create a dict for each position, store them as separate attributes
        graph = deepcopy(self.graph)
        nx.set_node_attributes(graph, values=pos, name='pos')
        nx.set_node_attributes(graph, values=node_colors, name='color')
            
        self.save_data(data=self.graph, data_type='graphml', subdir=True, file_name=f'nk-{self.info.as_string()}.graphml')
        # for node in graph.nodes():
        #     attributes = graph.nodes[node]
        #     print(f"Node {node} attributes: {attributes}")


    def visualize(self, plttitle=None):
        # Use pygraphviz for node positions, nx for visualizatoin
        
        # Store layouts because it takes a lot of time to calculate them
        pkl_path = self.get_file_path(file_name=f'pos-{self.info.as_string(omit=[self.attr])}.pkl', subdir=True) 
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                pos = pickle.load(f)
        else:
            start = time.time()
            pos = nx.nx_agraph.graphviz_layout(self.graph, self.prog) # dict
            print(f'{time.time()-start}s to get pgv layout.')

            with open(pkl_path, 'wb') as f:
                pickle.dump(pos, f)
        

        # Order the colors in according to the graph nodes
        node_colors = self.metadf['color'].to_dict()
        nodes = list(self.graph.nodes)
        node_colors = dict(sorted(node_colors.items(), key=lambda x: nodes.index(x[0])))
        assert list(node_colors.keys()) == nodes
        
        # pos = dict(sorted(pos.items(), key=lambda x: nodes.index(x[0])))
        # assert list(pos.keys()) == nodes
        # self.save_graphml(node_colors, pos)

        start = time.time()
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw_networkx(
            self.graph, 
            pos, 
            ax=ax, 
            node_size=10,
            width=0.1,
            with_labels=False,  
            node_color=list(node_colors.values()))
        
        if plttitle is not None:
            title = ', '.join([f'{col}: {val}' for col, val in plttitle.iloc[0].items()])
            plt.title(title, fontsize=8)


        ax.grid(False)
        
        print(f'{time.time()-start}s to draw nx plot.')
        self.save_data(data=plt, data_type=self.data_type, subdir=True, file_name=f'nk-{self.info.as_string()}.{self.data_type}')



    # def draw_pgv(self):

    #     def nx_to_pgv(graph):
    #         return nx.nx_agraph.to_agraph(graph)
        
    #     metadf = self.cm.get_color_map(pgv=True)
    #     start = time.time()
    #     graph = nx_to_pgv(self.network.graph)

    #     for node in graph.nodes():
    #         # graph.get_node(node).attr['color'] = self.colormap[node]
    #         # graph.get_node(node).attr['shape'] = self.shapemap[node]
    #         graph.get_node(node).attr['fillcolor'] = metadf.loc[node, 'color']
    #         graph.get_node(node).attr['style'] = 'filled'
    #         graph.get_node(node).attr['fixedsize'] = 'true'
    #         graph.get_node(node).attr['width'] = graph.get_node(node).attr['height'] = 1
    #         # graph.edge_attr['penwidth'] = 1.0 
    #         # graph.graph_attr['label'] = f'{self.attr}-{self.cluster_alg}-{self.language}'
    #         graph.get_node(node).attr['label'] = '' # Remove label


    #     file_path = self.get_file_path(file_name=f'network-{self.info.as_string()}.{self.data_type}', subdir=True)
    #     graph.draw(file_path, prog='fdp')
    #     # img = graph.draw(prog='fdp') 
    #     # display(Image(img))
    #     self.logger.info(f'Created pygraphviz grap.')
    #     print(f'{time.time()-start}s to produce pgv graph.')

    #     # for edge in graph.edges():
    #     #     source, target = edge  # Unpack the source and target nodes of the edge
    #     #     weight = graph.get_edge(source, target).attr['weight']  # Get the weight of the edge
    #     #     print(f"Edge from {source} to {target}, Weight: {weight}")

    #     # Show in IDE
    #     if self.data_type != 'svg':
    #         img = plt.imread(file_path)
    #         # Create a matplotlib figure and display the graph image
    #         fig, ax = plt.subplots(figsize=(6, 4))
    #         ax.axis('off')
    #         # ax.set_title(file_name)
    #         ax.imshow(img)
    #         # Show the graph in the IDE
    #         plt.show()

