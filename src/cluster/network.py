import logging
import numpy as np
import networkx as nx
import networkit as nk
import matplotlib.pyplot as plt
import random
import pandas as pd
import pygraphviz as pgv
import sys
sys.path.append("..")
from utils import DataHandler
from .cluster import NetworkCluster
from .cluster_utils import ColorMap, ShapeMap


class NXNetwork(DataHandler):
    def __init__(self, language, mx, cluster_alg=None, graph=None):
        super().__init__(language, output_dir='similarity', data_type='svg')

        self.mx = mx
        self.cluster_alg = cluster_alg
        self.graph = graph

        if (self.mx is not None and self.graph is not None) or (self.mx is None and self.graph is None):
            raise ValueError('Pass either matrix or graph.')
        if self.graph is None and self.mx is not None:
            self.graph = self.network_from_mx()


    def get_clusters(self):
        nc  = NetworkCluster(self.language, self.graph, self.cluster_alg)
        clusters = nc.cluster()
        return clusters


    def network_from_mx(self):
        if self.mx.mx.equals(self.mx.mx.T):
            graph = nx.from_pandas_adjacency(self.mx.mx)
            self.logger.warning(f'Created undirected graph from symmetric matrix.')
        else:
            graph = nx.from_pandas_adjacency(self.mx.mx, create_using=nx.DiGraph) 
            self.logger.warning(f'Created directed graph from non-symmetric matrix.')
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph
     

    def get_graph_info(self, graph):
        if isinstance(graph, nx.Graph):
            print(f'--------------\n'
                f'nx Graph Overview: \n'
                f'Nr. of nodes: {self.graph.number_of_nodes()} \n'
                f'Nr. of edges: {self.graph.number_of_edges()} \n'
                f'is weighted: {nx.is_weighted(self.graph)} \n'
                f'is directed: {nx.is_directed(self.graph)} \n'
                f'Nr. of Selfloops: {nx.number_of_selfloops(self.graph)} \n'
                f'--------------\n')
        elif isinstance(graph, nk.Graph):
            print('nk Graph Overview:', nk.overview(self.graph))
        elif isinstance(graph, pgv.AGraph):
            print(f'--------------\n'
                f'PGV Graph Overview\n'
                f'Number of Nodes: {graph.number_of_nodes()}\n'
                f'Number of Edges: {graph.number_of_edges()}\n'
                f'Is Directed: {graph.is_directed()}\n'
                f'Layout Program: {graph.layout()}\n'
                f'--------------\n')


    def nx_check_nan_edge_weights(self):
        '''
        Return True if a graph has nan edge weights.
        '''
        edge_labels = nx.get_edge_attributes(self.graph,'weight')
        weights = [weight for tup, weight in edge_labels.items()]
        nan_bool = np.any(np.isnan(weights)) # True if there are nan
        return nan_bool

    def replace_nan_edge_weights(self, replacement_value=0):
        '''
        graph: nx graph
        replacement_value: value with which np.nans should be replaced
        '''
        edge_labels = nx.get_edge_attributes(self.graph,'weight')
        new_edge_labels = {tup: replacement_value if np.isnan(weight) else weight for tup, weight in edge_labels.items()}
        
        nx.set_edge_attributes(self.graph, new_edge_labels, "weight") # inplace!
        # self.get_graph_info(self.graph)

    def nx_to_nk(self, graph):
        return nk.nxadapter.nx2nk(graph, weightAttr='weight') #weightAttr: The edge attribute which should be treated as the edge weight
    
    def nk_to_nx(self, graph):
        return nk.nxadapter.nk2nx(graph)      

    def nk_iterate_edges(self):
        def edgeFunc(u, v, weight, edgeId):
                print("Edge from {} to {} has weight {} and id {}".format(u, v, weight, edgeId))
        # Using iterator with callback function.
        self.graph.forEdges(edgeFunc) # forEdges iterator from nk

    def nk_complete_weighted_graph(self, n):
        graph = nx.complete_graph(n)
        graph = nk.nxadapter.nx2nk(graph)
        # Iterate over the edges of graph and add weights
        graph = nk.graphtools.toWeighted(graph) # weights = 1
        for u, v, w in graph.iterEdgesWeights():
                w = random.random()
                graph.setWeight(u, v, w)
        return graph

    def random_weighted_undirected_graph(self):
        nkgraph = nk.generators.MocnikGenerator(dim=2, n=100, k=2, weighted=True).generate() # Density parameter, determining the ratio of edges to nodes
        nxgraph = self.nk_to_nx(nkgraph).to_undirected()

        plt.figure(1, figsize=(11, 11))
        #nx.draw_networkx(nxgraph)
        #pos = nx.circular_layout(nxgraph)
        pos = nx.spring_layout(nxgraph)
        #nx.draw_networkx(nxgraph, pos=pos, node_size=30, with_labels=False, node_color='red')
        edge_labels = nx.get_edge_attributes(nxgraph,'weight')
        for currtuple, weight in edge_labels.items():
                edge_labels[currtuple] = round(weight, 3)
        nx.draw_networkx_nodes(nxgraph, pos, node_size=20, node_color='red')
        nx.draw_networkx_edges(nxgraph, pos, edge_color="blue", width=1)
        nx.draw_networkx_edge_labels(nxgraph, pos=pos, edge_labels = edge_labels, font_size=6, verticalalignment='top')
        plt.show()


class NetworkViz(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='similarity', data_type='svg')
        self.add_subdir()      
        
    def nx_to_pgv(self, graph):
        return nx.nx_agraph.to_agraph(graph)
        
    def draw_graph(self):
        self.logger.info(f'Drawing graph for {self.name_string}')
        self.colormap = self.get_colormap(cluster_list=self.clusters.clusters)
        self.shapemap = self.get_shapemap(cluster_list=None)
        self.draw_graph_pgv()

    
    def draw_graph_nx(self):
        # No graph drawing possible in Networkit
        pos = nx.spring_layout(self.graph, seed=8)
        nx.draw_networkx(self.graph, pos, node_size=30, with_labels=False, node_color=self.colormap)
        plt.show()

    def draw_graph_pgv(self):
        graph = self.nx_to_pgv(self.graph)
        self.get_graph_info(graph)
        for node in graph.nodes():
            # graph.get_node(node).attr['color'] = self.colormap[node]
            graph.get_node(node).attr['shape'] = self.shapemap[node]
            graph.get_node(node).attr['fillcolor'] = self.colormap[node]
            graph.get_node(node).attr['style'] = 'filled'
            graph.get_node(node).attr['fixedsize'] = 'true'
            graph.get_node(node).attr['width'] = graph.get_node(node).attr['height'] = 1
            # graph.edge_attr['penwidth'] = 1.0 
            # graph.graph_attr['label'] = f'{self.attr}-{self.cluster_alg}-{self.language}'
            graph.get_node(node).attr['label'] = '' # Remove label

        file_path = self.get_file_path(file_name=f'network-{self.name_string}.svg', dpi=600) ######################### Add language to file name?
        graph.draw(file_path, prog='fdp') #fdp
        self.logger.info(f'Created pygraphviz grap.')

        # for edge in graph.edges():
        #     source, target = edge  # Unpack the source and target nodes of the edge
        #     weight = graph.get_edge(source, target).attr['weight']  # Get the weight of the edge
        #     print(f"Edge from {source} to {target}, Weight: {weight}")

        # Show in IDE
        img = plt.imread(file_path)
        # Create a matplotlib figure and display the graph image
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')
        # ax.set_title(file_name)
        ax.imshow(img)
        # Show the graph in the IDE
        plt.show()

    def get_colormap(self, cluster_list=None):
        '''
        Map attribute names or clusters to color.
        If cluster_list is None, map attributes.
        If cluster_list is list of clusters, map cluster.
        '''
        if self.attr is not None or cluster_list is not None:
            fn_color_mapping = ColorMap(language=self.language, attr=None, cluster_list=cluster_list).get_color_map()
            # assert self.graph.number_of_nodes() == len(fn_color_mapping) ####################
        else:
            fn_color_mapping = {node: 'blue' for node in self.graph}
        return fn_color_mapping
    
    def get_shapemap(self, cluster_list):
        if cluster_list is None: ###################
            fn_shape_mapping = ShapeMap(self.language, self.attr, cluster_list).get_shape_map()
        else:
            fn_shape_mapping = {node: 'point' for node in self.graph}
        return fn_shape_mapping
