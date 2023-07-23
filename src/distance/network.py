
import numpy as np
import networkx as nx
import networkit as nk
import matplotlib.pyplot as plt
import random
import pandas as pd
import pygraphviz as pgv
from .distance_visualization import ColorMap
import sys
sys.path.append("..")
from utils import DataHandler


class Network(DataHandler):
    def __init__(self, language, name_mx_tup=(1, 2), graph=None, draw=True, cluster_alg=None, attribute_name=None): # attribute: labeled groups, i.e. 'm', 'f' #################################):
        super().__init__(language, 'distance', 'png')
        
        self.distname, self.mx = name_mx_tup
        print(f'Network based on {self.distname}')
        self.graph = graph
        # if (self.mx is not None and self.graph is not None) or (self.mx is None and self.graph is None):
        #     raise ValueError('Pass either matrix or graph.')
        if self.graph is None and self.mx is not None:
            self.graph = self.network_from_mx()
            self.get_graph_info(self.graph)

        self.draw = draw
        self.cluster_alg = cluster_alg
        self.attribute_name = attribute_name
        self.cluster_algs = ['louvain']
        self.attribute_names = ['gender', 'author', 'canon']


    def get_graph_info(self):
        raise NotImplementedError

    def get_colormap(self):
        if self.cluster_alg is not None or self.attribute_name is not None:
            if (self.attribute_name == 'author') or (self.attribute_name == 'canon') or (self.attribute_name == 'gender'):
                cluster_list = list(self.graph.nodes(data=False)) # get list of file names
            # else:
            #     cluster_list = self.clusters
            #     self.attribute_name = self.cluster_alg

            fn_color_mapping = ColorMap(
                self.attribute_name,
                cluster_list,
                self.language).get_color_map()
            # assert self.graph.number_of_nodes() == len(fn_color_mapping)##################################
            #color_map = [fn_color_mapping[node] for node in self.graph if node in fn_color_mapping] # if node in fn_color_mapping #############################3
        else:
            fn_color_mapping = {node: 'blue' for node in self.graph}
        return fn_color_mapping

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
        '''
        mx: Adjacency or directed or undirected weight matrix
        '''
        mx = self.mx.fillna(0) # Adjacency matrix must not contain Nan!
        if self.mx.equals(self.mx.T):
            graph = nx.from_pandas_adjacency(self.mx)
            print('Matrix is symmetric.')
        else:
            graph = nx.from_pandas_adjacency(self.mx, create_using=nx.DiGraph) 
            print('Matrix is not symmetric but directed.')
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph

    def create_clusters(self):
        if self.cluster_alg == 'louvain':
            c = nx.community.louvain_communities(self.graph, weight='weight', seed=11, resolution=0.1)

        if self.draw:
            self.colormap = self.get_colormap()
            # for k, v in self.colormap.items():
            #     print(k, v)
            self.draw_graph()
        
        return c
        
    def nx_to_pgv(self, graph):
        return nx.nx_agraph.to_agraph(graph)
        
    def draw_graph(self):
        self.logger.info(f'Drawing graph for {self.distname} {self.cluster_alg} {self.attribute_name}')
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
            graph.get_node(node).attr['label'] = node
            graph.get_node(node).attr['color'] = self.colormap[node]
            graph.get_node(node).attr['shape'] = 'point'
            graph.get_node(node).attr['fixedsize'] = 'true'
            graph.get_node(node).attr['width'] = graph.get_node(node).attr['height'] = 0.1
            # graph.edge_attr['penwidth'] = 1.0 
            # graph.graph_attr['label'] = f'{self.attribute_name}-{self.cluster_alg}-{self.language}'

        file_name = f'{self.distname}-{self.attribute_name}-{self.cluster_alg}-{self.language}'
        file_path = self.get_file_path(file_name=f'network-{file_name}.png', dpi=600)
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
        ax.set_title(file_name)
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
        self.get_graph_info(self.graph)

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
