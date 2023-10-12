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
from .cluster import SimmxCluster, NetworkCluster
from .network_viz import ColorMap, ShapeMap


class Network(DataHandler):
    def __init__(self, language, name_mx_tup=(1, 2), graph=None, draw=True, cluster_alg=None, attribute_name=None): # attribute: labeled groups, i.e. 'm', 'f' #################################):
        super().__init__(language, 'distance', 'svg')
        self.language = language
        self.distname, self.mx = name_mx_tup
        self.graph = graph
        # if (self.mx is not None and self.graph is not None) or (self.mx is None and self.graph is None):
        #     raise ValueError('Pass either matrix or graph.')
        if self.graph is None and self.mx is not None:
            self.graph = self.network_from_mx()
            # self.get_graph_info(self.graph)

        self.draw = draw
        self.cluster_alg = cluster_alg
        self.attribute_name = attribute_name

        self.simmx_cluster_algs = SimmxCluster(self.language).cluster_algs
        self.network_cluster_algs = NetworkCluster().cluster_algs
        self.attribute_names = ['gender', 'author']##################, 'canon']

        self.clusters = None


    def get_graph_info(self):
        raise NotImplementedError

    def get_colormap(self, cluster_list=None):
        '''
        Map attribute names or clusters to color.
        If cluster_list is None, map attributes.
        If cluster_list is list of clusters, map cluster.
        '''
        if self.attribute_name is not None or cluster_list is not None:
            fn_color_mapping = ColorMap(language=self.language, attribute_name=None, cluster_list=cluster_list).get_color_map()
            # assert self.graph.number_of_nodes() == len(fn_color_mapping) ####################
        else:
            fn_color_mapping = {node: 'blue' for node in self.graph}
        return fn_color_mapping
    
    def get_shapemap(self, cluster_list):
        if cluster_list is None: ###################
            fn_shape_mapping = ShapeMap(self.language, self.attribute_name, cluster_list).get_shape_map()
        else:
            fn_shape_mapping = {node: 'point' for node in self.graph}
        return fn_shape_mapping


    # def get_jazz_network(self):
    #     print('Jazz network as networkit grap.')
    #     # Jazz network (unweighted edge list)
    #     jazz_path = '/home/annina/scripts/great_unread_nlp/src/download.tsv.arenas-jazz/arenas-jazz/out.arenas-jazz'
    #     # col_names = ['source', 'target'] #see readme of dataset: 'weight', 'timestamp' should also exist but don't
    #     # df = pd.read_csv(jazz_path, sep='\t', header=0, names=col_names)
    #     # print(df)
    #     edgeListReader = nk.graphio.EdgeListReader('\t', 1, '%')
    #     graph = edgeListReader.read(jazz_path)
    #     return graph



class NXNetwork(Network):
    def __init__(self, language, name_mx_tup=(None, None), graph=None, draw=True, cluster_alg=None, attribute_name=None):
        super().__init__(language=language, name_mx_tup=name_mx_tup, graph=graph, draw=draw, cluster_alg=cluster_alg, attribute_name=attribute_name)

    def network_from_mx(self):
        if self.mx.equals(self.mx.T):
            graph = nx.from_pandas_adjacency(self.mx)
            print('Matrix is symmetric.')
        else:
            graph = nx.from_pandas_adjacency(self.mx, create_using=nx.DiGraph) 
            print('Matrix is not symmetric but directed.')
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph
    
    def get_clusters(self):
        if self.cluster_alg in self.network_cluster_algs:
            self.clusters = NetworkCluster(self.graph, self.attribute_name, self.cluster_alg)
            self.clusters.cluster()
            print(self.clusters)
        else:
            self.clusters = SimmxCluster(self.language, self.mx, self.attribute_name, self.cluster_alg)
            self.clusters.cluster()

        if self.draw:
            self.draw_graph()
        
        
    def nx_to_pgv(self, graph):
        return nx.nx_agraph.to_agraph(graph)
        
    def draw_graph(self):
        self.logger.info(f'Drawing graph for {self.distname} {self.cluster_alg} {self.attribute_name}')
        if self.cluster_alg in self.network_cluster_algs:
            self.colormap = self.get_colormap(cluster_list=self.clusters.clusters)
            self.shapemap = self.get_shapemap(cluster_list=None)
            self.draw_graph_pgv()
        elif self.cluster_alg in self.simmx_cluster_algs:
            self.draw_simmx()

    def draw_simmx(self):

        sorted_mxs = []
        for cluster in self.clusters.clusters:
            df = self.mx.loc[:, cluster].sort_index(axis=1)
            sorted_mxs.append(df)
        sorted_mx = pd.concat(sorted_mxs, axis=1)
        for i in range(0, len(sorted_mxs)):
            print(f'Cluster {i} has {sorted_mxs[i].shape[1]} elements.')

        sorted_mxs = []
        for cluster in self.clusters.clusters:
            df = sorted_mx.loc[cluster, :].sort_index(axis=0)
            sorted_mxs.append(df)
        sorted_mx = pd.concat(sorted_mxs, axis=0)

        assert self.mx.shape == sorted_mx.shape

        # Create the heatmap using matplotlib's imshow function
        # hot_r, viridis, plasma, inferno
        plt.imshow(sorted_mx, cmap='plasma', interpolation='nearest')

        # Add a color bar to the heatmap for better understanding of the similarity values
        plt.colorbar()

        # Add axis labels and title (optional)
        # plt.xlabel('Data Points')
        # plt.ylabel('Data Points')

        title = [f"{i}: nr-elements: {round(len(self.clusters.clusters[i]), 2)}, mf-ratio: {round(self.clusters.qual[i], 2)}" for i in range(len(self.clusters.clusters))]
        title = ('\n').join(title)
        plt.title(title, fontsize=8)

        # Show the heatmap
        #
        # plt.show()
        self.save_data(data=plt, data_type='svg', file_name=f'heatmap-{self.distname}{self.attribute_name}-{self.cluster_alg}.svg')
        plt.close()

    
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
            # graph.graph_attr['label'] = f'{self.attribute_name}-{self.cluster_alg}-{self.language}'
            graph.get_node(node).attr['label'] = '' # Remove label

        file_name = f'{self.distname}-{self.attribute_name}-{self.cluster_alg}-{self.language}'
        file_path = self.get_file_path(file_name=f'network-{file_name}.svg', dpi=600)
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
