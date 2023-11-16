
import pandas as pd
import itertools
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pickle
from copy import deepcopy
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
from .cluster_utils import ColorMap
import logging
logging.basicConfig(level=logging.DEBUG)

class MxReorder():
    '''Sort row and column indices so that clusters are visible in heatmap.'''

    ORDERS = ['fn', 'olo']

    def __init__(self, language, mx, info, metadf):
        self.language = language
        self.mx = mx
        self.info = info
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
        # Sort rows and columns of each cluster according to file name, which starts with the name of the author
        ordmxs = []

        # Get index labels belonging to the current cluster
        for cluster in self.metadf['cluster'].unique():
            file_names = self.metadf[self.metadf['cluster'] == cluster].index.tolist()

            df = self.mx.mx.loc[:, file_names].sort_index(axis=1)
            ordmxs.append(df)
        ordmx = pd.concat(ordmxs, axis=1)

        ordmxs = []
        for cluster in self.metadf['cluster'].unique():
            file_names = self.metadf[self.metadf['cluster'] == cluster].index.tolist()
            df = ordmx.loc[file_names, :].sort_index(axis=0)
            ordmxs.append(df)
        ordmx = pd.concat(ordmxs, axis=0)
        return ordmx

    
    def order_olo(self):
        # Use distance matrix for linkage function

        sorted_fns = []
        # Get unique cluster lables, sorted from rarest to most common label
        unique_clust = self.metadf['cluster'].value_counts().sort_values().index.tolist()

        # Iterate over unique clusters
        for cluster in unique_clust:

            # Extract file names for the current cluster
            file_names = self.metadf[self.metadf['cluster'] == cluster].index.tolist()

            # If or 2 elements in cluster, order doesn't matter
            if len(file_names) <=2:
                sorted_fns.extend(file_names)

            # Get OLO for cur rent cluster
            else:
                # Subset the similarity matrix for the current cluster
                cmx = self.mx.dmx.loc[file_names, file_names]
                sq_cmx = squareform(cmx)
                
                cluster_linkage = linkage(sq_cmx, method='average')
                
                order = leaves_list(optimal_leaf_ordering(cluster_linkage, sq_cmx))
                # Map integer indices to string indices
                ordered_fn = cmx.index[order].tolist()
                sorted_fns.extend(ordered_fn)
        


        assert len(set(sorted_fns)) == len(sorted_fns)
        ordmx = self.mx.dmx.loc[sorted_fns, sorted_fns]

        # Convert back to similarity
        ordmx = SimMx(self.language, name='olo', mx=ordmx, normalized=True, is_sim=False, is_directed = self.mx.is_directed, is_condensed=False, has_dmx=True)

        nr_texts = DataHandler(self.language).nr_texts
        assert (ordmx.mx.shape[0] == nr_texts) and (ordmx.mx.shape[1] == nr_texts) 

        # Reorder the final matrix based on the optimal order of clusters ############################

        self.logger.info(f'OLO matrix reorder.')
        return ordmx


class MxViz(DataHandler):
    def __init__(self, language, mx, clusters, info, param_comb, metadf):
        super().__init__(language, output_dir='similarity')
        self.mx = mx
        self.clusters = clusters
        self.info = info
        self.param_comb = param_comb
        self.metadf = metadf
        self.attr=self.info.attr
        self.add_subdir('mxviz')


    def draw_heatmap(self, qual):
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

        title = ', '.join([f'{col}: {val}' for col, val in qual.iloc[0].items()])

        plt.title(title, fontsize=8)

        self.save_data(data=plt, data_type='png', subdir=True, file_name=f'heatmap-{self.info.as_string()}.png')
        plt.close()


    def draw_logreg(self, feature, model, X, y_true, y_pred):
        # Visualize the decision boundary
        plt.figure(figsize=(10, 6))
        plt.grid(True)

        # Generate a range of values for X for plotting the decision boundary
        X_range = np.linspace(min(X), max(X), 300).reshape(-1, 1)
        # Predict the corresponding y values for the X_range
        y_range = model.predict(X_range)

        # Plot the decision boundary
        plt.plot(X_range, y_range, color='red', linewidth=3, label='Decision Boundary')

        # Scatter plot for the data points
        plt.scatter(X, y_true, c=y_pred, cmap='Set1', edgecolors='k', marker='o', s=100, label='Clusters (logreg)')

        # Set labels and title
        plt.xlabel(f'{feature.capitalize()}')
        plt.ylabel('Clusters (Cluster Alg)')
        plt.title('Logistic Regression')

        plt.yticks(np.unique(y_true))

        # Display the legend
        plt.legend()
        self.save_data(data=plt, data_type='png', subdir=True, file_name=f'logreg-{self.info.as_string()}-{feature}.png')
        plt.close()
    


class NkViz(DataHandler):
    PROGS = ['fdp']

    def __init__(self, language, network, info, metadf):
        super().__init__(language, output_dir='similarity', data_type='svg')
        self.network = network
        self.graph = self.network.graph
        print(network.get_graph_info())
        self.info = info
        self.attr=self.info.attr
        self.metadf = metadf

        self.cm = ColorMap(self.metadf)
        self.add_subdir('nkviz')
        self.logger.info(f'Drawing graph for {self.info.as_string()}')
     

    def save_graphml(self, node_colors, pos):
        # for k, v in node_colors.items():
        #     print(type(k), type(v))
        # for k, v in pos.items():
        #     print(type(k), type(v))
        nx.set_node_attributes(self.graph, values=node_colors, name='color')
        nx.set_node_attributes(self.graph, values=pos, name='pos')

        for node, attributes in nx.get_node_attributes(self.graph, 'all').items():
            print('print node attrs')
            print(f"Node {node}: {attributes}")
            print('#######################################')
            
        # self.save_data(data=self.graph, data_type='graphml', subdir=True, file_name=f'nk-{self.info.as_string()}.graphml')


    def draw_nx(self):
        # Use pygraphviz for node positions
        metadf = self.cm.get_color_map()

        pkl_path = self.get_file_path(file_name=f'pos-{self.info.as_string()}.pkl', subdir=True) 
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                pos = pickle.load(f)
        else:
            start = time.time()
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog='fdp') # dict
            print(f'{time.time()-start}s to get pgv layout.')

            with open(pkl_path, 'wb') as f:
                pickle.dump(pos, f)
        

        # Order the colors in according to the graph nodes
        node_colors = metadf['color'].to_dict()
        nodes = list(self.graph.nodes)
        node_colors = dict(sorted(node_colors.items(), key=lambda x: nodes.index(x[0])))
        assert list(node_colors.keys()) == nodes
        self.save_graphml(node_colors, pos)

        start = time.time()
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw_networkx(
            self.graph, 
            pos, 
            ax=ax, 
            node_size=30,
            width=0.1,
            with_labels=False,  
            node_color=list(node_colors.values()))
        

        ax.grid(False)
        
        print(f'{time.time()-start}s to draw nx plot.')
        self.save_data(data=plt, data_type=self.data_type, subdir=True, file_name=f'nk-{self.info.as_string()}.{self.data_type}')



    def draw_pgv(self):

        def nx_to_pgv(graph):
            return nx.nx_agraph.to_agraph(graph)
        
        metadf = self.cm.get_color_map(pgv=True)
        start = time.time()
        graph = nx_to_pgv(self.network.graph)

        for node in graph.nodes():
            # graph.get_node(node).attr['color'] = self.colormap[node]
            # graph.get_node(node).attr['shape'] = self.shapemap[node]
            graph.get_node(node).attr['fillcolor'] = metadf.loc[node, 'color']
            graph.get_node(node).attr['style'] = 'filled'
            graph.get_node(node).attr['fixedsize'] = 'true'
            graph.get_node(node).attr['width'] = graph.get_node(node).attr['height'] = 1
            # graph.edge_attr['penwidth'] = 1.0 
            # graph.graph_attr['label'] = f'{self.attr}-{self.cluster_alg}-{self.language}'
            graph.get_node(node).attr['label'] = '' # Remove label


        file_path = self.get_file_path(file_name=f'network-{self.info.as_string()}.{self.data_type}', subdir=True)
        graph.draw(file_path, prog='fdp')
        # img = graph.draw(prog='fdp') 
        # display(Image(img))
        self.logger.info(f'Created pygraphviz grap.')
        print(f'{time.time()-start}s to produce pgv graph.')

        # for edge in graph.edges():
        #     source, target = edge  # Unpack the source and target nodes of the edge
        #     weight = graph.get_edge(source, target).attr['weight']  # Get the weight of the edge
        #     print(f"Edge from {source} to {target}, Weight: {weight}")

        # Show in IDE
        if self.data_type != 'svg':
            img = plt.imread(file_path)
            # Create a matplotlib figure and display the graph image
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axis('off')
            # ax.set_title(file_name)
            ax.imshow(img)
            # Show the graph in the IDE
            plt.show()

